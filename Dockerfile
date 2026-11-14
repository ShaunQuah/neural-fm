# CPU-only image: runs the test suite, or re-scores the bundled checkpoint on MC_Maze.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Dependencies first so this layer is cached when only the source changes.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install numpy scipy pytest

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps -e .

COPY tests ./tests
COPY models ./models

# Default: run the tests (needs no data). For the evaluation, mount the processed dataset:
#   docker run --rm -v <folder with mc_maze_5ms.npz>:/data neural-fm \
#       python -m neuralfm.evaluate --cache /data/mc_maze_5ms.npz
CMD ["pytest", "-q", "-rs"]
