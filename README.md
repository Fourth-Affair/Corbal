# Next.js + FastAPI starter

This repository runs a Next.js frontend and a FastAPI backend with Docker Compose.

## Start the app

```sh
docker compose up --build
```

Open [http://localhost:3000](http://localhost:3000) for the frontend. The API is available at [http://localhost:8000](http://localhost:8000), with interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

The frontend and backend source folders are mounted into their containers for development. Both services reload when their source changes. Rebuild the services after changing their dependencies:

```sh
docker compose up --build
```

Stop the stack with `Ctrl+C`, or run `docker compose down` in another terminal.
