FROM docker.io/library/node:22-bookworm-slim AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci

# Development (compose.yaml): source is bind-mounted, `npm run dev` with hot reload
FROM deps AS dev
COPY . .
EXPOSE 3000
CMD ["npm", "run", "dev"]

# Production build (compose.prod.yaml). NEXT_PUBLIC_* values are inlined by `next build`, so they are build args.
FROM deps AS build
ARG NEXT_PUBLIC_SUPABASE_URL=""
ARG NEXT_PUBLIC_SUPABASE_ANON_KEY=""
ARG NEXT_PUBLIC_SUPABASE_BUCKET="plans"
ENV NEXT_PUBLIC_SUPABASE_URL=$NEXT_PUBLIC_SUPABASE_URL \
    NEXT_PUBLIC_SUPABASE_ANON_KEY=$NEXT_PUBLIC_SUPABASE_ANON_KEY \
    NEXT_PUBLIC_SUPABASE_BUCKET=$NEXT_PUBLIC_SUPABASE_BUCKET \
    NEXT_TELEMETRY_DISABLED=1
COPY . .
RUN npm run build && npm prune --omit=dev

FROM docker.io/library/node:22-bookworm-slim AS prod
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
COPY --from=build --chown=node:node /app/package.json ./
COPY --from=build --chown=node:node /app/node_modules ./node_modules
COPY --from=build --chown=node:node /app/.next ./.next
USER node
EXPOSE 3000
CMD ["npm", "start"]
