"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    logger.info(
        "application_starting",
        app=settings.app_name,
        env=settings.app_env,
        version=__version__,
    )
    yield
    logger.info("application_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Production-style conversational AI guardrail service. "
            "Moderates assistant responses using local OSS models or "
            "cloud LLM + RAG policy retrieval."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "service": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


app = create_app()
