from __future__ import annotations

import io
import os
import secrets
import statistics
from typing import Any

import chess
import chess.engine
import chess.pgn
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field


PUBLIC_SERVER_URL = os.getenv(
    "PUBLIC_SERVER_URL",
    "https://chess-analysis-chatgpt.onrender.com",
)

app = FastAPI(
    title="Stockfish Chess Analysis API",
    description=(
        "Analyze chess positions, individual PGNs, and small batches of PGNs "
        "with Stockfish."
    ),
    version="2.0.0",
    servers=[
        {
            "url": PUBLIC_SERVER_URL,
            "description": "Production Stockfish server",
        }
    ],
)

security = HTTPBearer(auto_error=False)

STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "/usr/games/stockfish")
ENGINE_API_KEY = os.getenv("ENGINE_API_KEY", "")


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> None:
    if not ENGINE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="ENGINE_API_KEY is not configured on the server.",
        )

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token.")

    if not secrets.compare_digest(credentials.credentials, ENGINE_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key.")


class PositionRequest(BaseModel):
    fen: str
    time_ms: int = Field(default=1500, ge=50, le=10000)
    multipv: int = Field(default=3, ge=1, le=5)
    pv_plies: int = Field(default=14, ge=1, le=30)


class GameRequest(BaseModel):
    pgn: str
    time_ms_per_position: int = Field(default=100, ge=25, le=500)
    top_n: int = Field(default=8, ge=1, le=20)
    max_plies: int = Field(default=160, ge=10, le=300)


class BatchRequest(BaseModel):
    pgn_bundle: str
    target_player: str | None = None
    starting_game_number: int = Field(default=1, ge=1, le=100000)
    nodes_per_position: int = Field(default=2500, ge=500, le=20000)
    top_n_per_game: int = Field(default=3, ge=1, le=8)
    max_games: int = Field(default=5, ge=1, le=10)
    max_plies_per_game: int = Field(default=200, ge=10, le=400)


def open_engine() -> chess.engine.SimpleEngine:
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine.configure({"Threads": 1, "Hash": 64})
        return engine
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not start Stockfish: {exc}",
        ) from exc


def score_for_white(score: chess.engine.PovScore) -> int:
    value = score.pov(chess.WHITE)
    scored = value.score(mate_score=100000)
    return int(scored if scored is not None else 0)


def classify_loss(loss_cp: int) -> str:
    if loss_cp >= 300:
        return "blunder"
    if loss_cp >= 150:
        return "mistake"
    if loss_cp >= 75:
        return "inaccuracy"
    if loss_cp >= 30:
        return "small concession"
    return "normal"


def terminal_evaluation(board: chess.Board) -> int:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0
    return 100000 if outcome.winner == chess.WHITE else -100000


def summarize_player(
    evaluations: list[dict[str, Any]],
    player: str,
) -> dict[str, Any]:
    player_moves = [item for item in evaluations if item["player"] == player]
    non_forced = [item for item in player_moves if not item["forced"]]
    sample = non_forced or player_moves

    # A mating swing should count as a blunder, but allowing a single mate score
    # to make ACPL equal 40,000 is not useful. Cap each loss at ten pawns.
    capped_losses = [min(int(item["loss_cp"]), 1000) for item in sample]

    if capped_losses:
        average_loss = round(sum(capped_losses) / len(capped_losses), 1)
        median_loss = round(float(statistics.median(capped_losses)), 1)
        good_rate = round(
            100.0
            * sum(loss <= 30 for loss in capped_losses)
            / len(capped_losses),
            1,
        )
    else:
        average_loss = 0.0
        median_loss = 0.0
        good_rate = 0.0

    return {
        "player": player,
        "moves_analyzed": len(player_moves),
        "non_forced_decisions": len(non_forced),
        "average_loss_cp_capped": average_loss,
        "median_loss_cp_capped": median_loss,
        "moves_with_loss_30cp_or_less_percent": good_rate,
        "inaccuracies": sum(
            item["classification"] == "inaccuracy" for item in non_forced
        ),
        "mistakes": sum(
            item["classification"] == "mistake" for item in non_forced
        ),
        "blunders": sum(
            item["classification"] == "blunder" for item in non_forced
        ),
    }


def analyze_game_with_engine(
    engine: chess.engine.SimpleEngine,
    game: chess.pgn.Game,
    *,
    limit: chess.engine.Limit,
    top_n: int,
    max_plies: int,
    include_history: bool,
) -> dict[str, Any]:
    board = game.board()
    evaluations: list[dict[str, Any]] = []
    opening_moves: list[str] = []

    initial_info = engine.analyse(board, limit)
    previous_eval = score_for_white(initial_info["score"])

    for ply, move in enumerate(game.mainline_moves(), start=1):
        if ply > max_plies:
            break

        mover = board.turn
        player = "White" if mover == chess.WHITE else "Black"
        forced = board.legal_moves.count() <= 1
        move_san = board.san(move)

        if ply <= 20:
            opening_moves.append(move_san)

        fen_before = board.fen()
        eval_before = previous_eval
        board.push(move)
        fen_after = board.fen()

        if board.is_game_over(claim_draw=True):
            eval_after = terminal_evaluation(board)
        else:
            info = engine.analyse(board, limit)
            eval_after = score_for_white(info["score"])

        if mover == chess.WHITE:
            loss_cp = max(0, eval_before - eval_after)
        else:
            loss_cp = max(0, eval_after - eval_before)

        evaluations.append(
            {
                "ply": ply,
                "move_number": (ply + 1) // 2,
                "player": player,
                "move_san": move_san,
                "evaluation_before_cp_white": eval_before,
                "evaluation_after_cp_white": eval_after,
                "loss_cp": int(loss_cp),
                "classification": classify_loss(int(loss_cp)),
                "forced": forced,
                "fen_before": fen_before,
                "fen_after": fen_after,
            }
        )
        previous_eval = eval_after

    top_errors = sorted(
        [item for item in evaluations if not item["forced"]],
        key=lambda item: item["loss_cp"],
        reverse=True,
    )[:top_n]

    result: dict[str, Any] = {
        "headers": dict(game.headers),
        "result": game.headers.get("Result", "*"),
        "positions_analyzed": len(evaluations),
        "opening_moves_san": " ".join(opening_moves),
        "white_summary": summarize_player(evaluations, "White"),
        "black_summary": summarize_player(evaluations, "Black"),
        "top_errors": top_errors,
    }

    if include_history:
        result["evaluation_history"] = [
            {
                "ply": item["ply"],
                "move_number": item["move_number"],
                "player": item["player"],
                "move_san": item["move_san"],
                "evaluation_after_cp_white": item["evaluation_after_cp_white"],
            }
            for item in evaluations
        ]

    return result


def find_target_side(
    headers: dict[str, str],
    target_player: str | None,
) -> str:
    if not target_player:
        return ""

    target = target_player.strip().casefold()
    white = headers.get("White", "").strip().casefold()
    black = headers.get("Black", "").strip().casefold()

    if target == white:
        return "White"
    if target == black:
        return "Black"
    return ""



@app.get(
    "/health",
    operation_id="checkChessEngineHealth",
    dependencies=[Security(require_api_key)],
)
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "stockfish_path": STOCKFISH_PATH,
        "api_version": "2.0.0",
    }

@app.post(
    "/analyze-position",
    operation_id="analyzeChessPosition",
    dependencies=[Security(require_api_key)],
)
def analyze_position(request: PositionRequest) -> dict[str, Any]:
    try:
        board = chess.Board(request.fen)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {exc}") from exc

    if board.is_game_over(claim_draw=True):
        return {
            "fen": board.fen(),
            "side_to_move": "White" if board.turn else "Black",
            "analysis_time_ms": request.time_ms,
            "multipv": request.multipv,
            "game_over": True,
            "outcome": str(board.outcome(claim_draw=True)),
            "lines": [],
        }

    engine = open_engine()
    try:
        analysis = engine.analyse(
            board,
            chess.engine.Limit(time=request.time_ms / 1000.0),
            multipv=request.multipv,
        )
    finally:
        engine.quit()

    lines: list[dict[str, Any]] = []
    for rank, info in enumerate(analysis, start=1):
        pv = info.get("pv", [])[: request.pv_plies]
        score = info["score"].pov(chess.WHITE)
        lines.append(
            {
                "rank": rank,
                "evaluation_cp_white": score.score(mate_score=100000),
                "mate_for_white": score.mate(),
                "best_move_uci": pv[0].uci() if pv else "",
                "best_move_san": board.san(pv[0]) if pv else "",
                "principal_variation_uci": [move.uci() for move in pv],
                "principal_variation_san": board.variation_san(pv) if pv else "",
                "depth": int(info.get("depth", 0)),
                "nodes": int(info.get("nodes", 0)),
            }
        )

    return {
        "fen": board.fen(),
        "side_to_move": "White" if board.turn else "Black",
        "analysis_time_ms": request.time_ms,
        "multipv": request.multipv,
        "game_over": False,
        "outcome": "",
        "lines": lines,
    }


@app.post(
    "/analyze-game",
    operation_id="analyzeChessGame",
    dependencies=[Security(require_api_key)],
)
def analyze_game(request: GameRequest) -> dict[str, Any]:
    try:
        game = chess.pgn.read_game(io.StringIO(request.pgn))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid PGN: {exc}") from exc

    if game is None:
        raise HTTPException(status_code=400, detail="No chess game found in PGN.")

    engine = open_engine()
    try:
        result = analyze_game_with_engine(
            engine,
            game,
            limit=chess.engine.Limit(
                time=request.time_ms_per_position / 1000.0
            ),
            top_n=request.top_n,
            max_plies=request.max_plies,
            include_history=True,
        )
    finally:
        engine.quit()

    result["time_ms_per_position"] = request.time_ms_per_position
    return result


@app.post(
    "/analyze-batch",
    operation_id="analyzeChessBatch",
    dependencies=[Security(require_api_key)],
)
def analyze_batch(request: BatchRequest) -> dict[str, Any]:
    stream = io.StringIO(request.pgn_bundle)
    games: list[chess.pgn.Game] = []

    try:
        while len(games) < request.max_games:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            games.append(game)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse PGN batch: {exc}",
        ) from exc

    if not games:
        raise HTTPException(
            status_code=400,
            detail="No chess games were found in the PGN batch.",
        )

    engine = open_engine()
    results: list[dict[str, Any]] = []

    try:
        for offset, game in enumerate(games):
            analyzed = analyze_game_with_engine(
                engine,
                game,
                limit=chess.engine.Limit(nodes=request.nodes_per_position),
                top_n=request.top_n_per_game,
                max_plies=request.max_plies_per_game,
                include_history=False,
            )

            headers = analyzed["headers"]
            side = find_target_side(headers, request.target_player)

            if side == "White":
                target_summary = analyzed["white_summary"]
            elif side == "Black":
                target_summary = analyzed["black_summary"]
            else:
                target_summary = {
                    "player": "",
                    "moves_analyzed": 0,
                    "non_forced_decisions": 0,
                    "average_loss_cp_capped": 0.0,
                    "median_loss_cp_capped": 0.0,
                    "moves_with_loss_30cp_or_less_percent": 0.0,
                    "inaccuracies": 0,
                    "mistakes": 0,
                    "blunders": 0,
                }

            target_errors = [
                item for item in analyzed["top_errors"] if item["player"] == side
            ]

            results.append(
                {
                    "game_number": request.starting_game_number + offset,
                    "headers": headers,
                    "result": analyzed["result"],
                    "positions_analyzed": analyzed["positions_analyzed"],
                    "opening_moves_san": analyzed["opening_moves_san"],
                    "white_summary": analyzed["white_summary"],
                    "black_summary": analyzed["black_summary"],
                    "target_player": request.target_player or "",
                    "target_side": side,
                    "target_summary": target_summary,
                    "target_top_errors": target_errors,
                    "top_errors": analyzed["top_errors"],
                }
            )
    finally:
        engine.quit()

    return {
        "games_analyzed": len(results),
        "starting_game_number": request.starting_game_number,
        "nodes_per_position": request.nodes_per_position,
        "target_player": request.target_player or "",
        "results": results,
    }
