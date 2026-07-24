from __future__ import annotations

import io
import os
import secrets
from typing import Any

import chess
import chess.engine
import chess.pgn
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field


app = FastAPI(
    title="Stockfish Chess Analysis API",
    description="Analyze chess positions and PGN games with Stockfish.",
    version="1.0.0",
)

security = HTTPBearer(auto_error=False)

STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "/usr/games/stockfish")
ENGINE_API_KEY = os.getenv("ENGINE_API_KEY", "")


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> None:
    """Reject requests that do not contain the correct bearer token."""

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


def open_engine() -> chess.engine.SimpleEngine:
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine.configure(
            {
                "Threads": 1,
                "Hash": 64,
            }
        )
        return engine
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not start Stockfish: {exc}",
        ) from exc


def white_score(score: chess.engine.PovScore) -> dict[str, int | None]:
    """
    Return the evaluation from White's perspective.

    Positive centipawns favor White.
    Negative centipawns favor Black.
    """

    value = score.pov(chess.WHITE)

    return {
        "centipawns": value.score(mate_score=100000),
        "mate": value.mate(),
    }


def classify_loss(loss_cp: int) -> str:
    """Human-friendly, approximate classification."""

    if loss_cp >= 500:
        return "large blunder"
    if loss_cp >= 300:
        return "blunder "
    if loss_cp >= 100:
        return "small mistake"
    if loss_cp >= 50:
        return "small concession"
    return "normal"


@app.get("/health", operation_id="checkChessEngineHealth")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "stockfish_path": STOCKFISH_PATH,
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
        outcome = board.outcome(claim_draw=True)

        return {
            "fen": board.fen(),
            "game_over": True,
            "outcome": str(outcome),
            "lines": [],
        }

    engine = open_engine()

    try:
        analysis = engine.analyse(
            board,
            chess.engine.Limit(time=request.time_ms / 1000),
            multipv=request.multipv,
        )
    finally:
        engine.quit()

    lines: list[dict[str, Any]] = []

    for rank, info in enumerate(analysis, start=1):
        pv = info.get("pv", [])[: request.pv_plies]
        score = white_score(info["score"])

        lines.append(
            {
                "rank": rank,
                "evaluation_cp_white": score["centipawns"],
                "mate_for_white": score["mate"],
                "best_move_uci": pv[0].uci() if pv else None,
                "best_move_san": board.san(pv[0]) if pv else None,
                "principal_variation_uci": [move.uci() for move in pv],
                "principal_variation_san": (
                    board.variation_san(pv) if pv else ""
                ),
                "depth": info.get("depth"),
                "nodes": info.get("nodes"),
            }
        )

    return {
        "fen": board.fen(),
        "side_to_move": "White" if board.turn == chess.WHITE else "Black",
        "analysis_time_ms": request.time_ms,
        "multipv": request.multipv,
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

    board = game.board()
    engine = open_engine()

    evaluations: list[dict[str, Any]] = []

    try:
        initial_info = engine.analyse(
            board,
            chess.engine.Limit(time=request.time_ms_per_position / 1000),
        )
        previous_eval = white_score(initial_info["score"])["centipawns"]

        if previous_eval is None:
            previous_eval = 0

        for ply, move in enumerate(game.mainline_moves(), start=1):
            if ply > request.max_plies:
                break

            mover = board.turn
            player = "White" if mover == chess.WHITE else "Black"
            move_san = board.san(move)
            fen_before = board.fen()
            eval_before = previous_eval

            board.push(move)
            fen_after = board.fen()

            if board.is_game_over(claim_draw=True):
                outcome = board.outcome(claim_draw=True)

                if outcome is None or outcome.winner is None:
                    eval_after = 0
                elif outcome.winner == chess.WHITE:
                    eval_after = 100000
                else:
                    eval_after = -100000
            else:
                info = engine.analyse(
                    board,
                    chess.engine.Limit(
                        time=request.time_ms_per_position / 1000
                    ),
                )
                scored = white_score(info["score"])
                eval_after = scored["centipawns"]

                if eval_after is None:
                    eval_after = eval_before

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
                    "loss_cp": loss_cp,
                    "classification": classify_loss(loss_cp),
                    "fen_before": fen_before,
                    "fen_after": fen_after,
                }
            )

            previous_eval = eval_after

    finally:
        engine.quit()

    top_errors = sorted(
        evaluations,
        key=lambda item: item["loss_cp"],
        reverse=True,
    )[: request.top_n]

    return {
        "headers": dict(game.headers),
        "result": game.headers.get("Result", "*"),
        "positions_analyzed": len(evaluations),
        "time_ms_per_position": request.time_ms_per_position,
        "top_errors": top_errors,
        "evaluation_history": [
            {
                "ply": item["ply"],
                "move_number": item["move_number"],
                "player": item["player"],
                "move_san": item["move_san"],
                "evaluation_after_cp_white": item[
                    "evaluation_after_cp_white"
                ],
            }
            for item in evaluations
        ],
    }
