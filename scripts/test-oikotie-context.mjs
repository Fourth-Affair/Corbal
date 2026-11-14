#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DEFAULT_URL =
  "https://asunnot.oikotie.fi/myytavat-asunnot/helsinki/24784966";
const API_URL = "https://api.context.dev/v1/web/scrape";
const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function readOption(name) {
  const index = process.argv.indexOf(name);
  return index === -1 ? undefined : process.argv[index + 1];
}

function parseEnv(contents) {
  const values = new Map();
  for (const line of contents.split(/\r?\n/)) {
    const match = line.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/);
    if (!match) continue;

    let value = match[2].trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    } else {
      value = value.replace(/\s+#.*$/, "").trim();
    }
    values.set(match[1], value);
  }
  return values;
}

async function loadApiKey() {
  if (process.env.CONTEXT_DEV_API_KEY) return process.env.CONTEXT_DEV_API_KEY;

  for (const filename of [".env.local", ".env"]) {
    try {
      const contents = await readFile(path.join(projectRoot, filename), "utf8");
      const key = parseEnv(contents).get("CONTEXT_DEV_API_KEY");
      if (key) return key;
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
  }
  return undefined;
}

function validateListingUrl(rawUrl) {
  let url;
  try {
    url = new URL(rawUrl);
  } catch {
    throw new Error(`Invalid URL: ${rawUrl}`);
  }

  if (
    url.protocol !== "https:" ||
    url.hostname !== "asunnot.oikotie.fi" ||
    !url.pathname.startsWith("/myytavat-asunnot/")
  ) {
    throw new Error(
      "Use one Oikotie sale listing URL, such as https://asunnot.oikotie.fi/myytavat-asunnot/helsinki/24784966",
    );
  }
  return url.toString();
}

const schema = {
  type: "object",
  properties: {
    listing_title: {
      type: ["string", "null"],
      description: "The listing headline or apartment address, if shown.",
    },
    debt_free_price_eur: {
      type: ["number", "null"],
      description: "Velaton hinta in euros, not the monthly charge.",
    },
    sale_price_eur: {
      type: ["number", "null"],
      description: "Myyntihinta in euros, if separately stated.",
    },
    living_area_m2: {
      type: ["number", "null"],
      description: "Asuinpinta-ala in square metres.",
    },
    rooms: {
      type: ["string", "null"],
      description: "Huoneita, preserving the listing's notation such as 3h+k.",
    },
    floor: {
      type: ["string", "null"],
      description: "The apartment floor and total floors, e.g. 3 / 6.",
    },
    building_year: {
      type: ["integer", "null"],
      description: "Rakennusvuosi as a four-digit year.",
    },
    building_type: {
      type: ["string", "null"],
      description: "Rakennuksen tyyppi, e.g. kerrostalo.",
    },
    neighborhood: {
      type: ["string", "null"],
      description: "Kaupunginosa or neighborhood, e.g. Etu-Töölö.",
    },
    city: {
      type: ["string", "null"],
      description: "City, e.g. Helsinki.",
    },
    maintenance_charge_eur_per_month: {
      type: ["number", "null"],
      description: "Hoitovastike in euros per month (€/kk).",
    },
  },
  required: [
    "listing_title",
    "debt_free_price_eur",
    "sale_price_eur",
    "living_area_m2",
    "rooms",
    "floor",
    "building_year",
    "building_type",
    "neighborhood",
    "city",
    "maintenance_charge_eur_per_month",
  ],
  additionalProperties: false,
};

function isFloorplanImage(image) {
  const searchableText = `${image.alt ?? ""} ${image.url ?? ""}`.toLocaleLowerCase("fi-FI");
  return /pohjakuva|pohjapiir|pohja[-_ ]?kuva|floor[-_ ]?plan|floorplan|layout/.test(
    searchableText,
  );
}

async function main() {
  const rawUrl = readOption("--url") ?? DEFAULT_URL;
  const listingUrl = validateListingUrl(rawUrl);
  const apiKey = await loadApiKey();

  if (!apiKey) {
    console.error("Missing CONTEXT_DEV_API_KEY. Add it to the project .env file first.");
    process.exitCode = 1;
    return;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 90_000);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        url: listingUrl,
        formats: { markdown: true, json: true, images: true },
        jsonParams: {
          schema,
          instructions:
            "Extract only facts stated on this Oikotie apartment listing. Do not infer missing values. Prices and charges must be plain numbers in euros. Use null when a fact is absent. Keep the room and floor notation as shown.",
        },
        maxAgeMs: 0,
        timeoutOpts: { milliseconds: 85_000, behavior: "return-partial" },
      }),
      signal: controller.signal,
    });

    const result = await response.json().catch(() => null);
    if (!response.ok) {
      const message = result?.message ?? result?.error?.message ?? `HTTP ${response.status}`;
      throw new Error(`Context.dev request failed: ${message}`);
    }

    const images = result?.images?.success ? result.images.data ?? [] : [];
    const floorplanCandidates = images.filter(isFloorplanImage).map((image) => ({
      url: image.url,
      alt: image.alt ?? null,
      width: image.width ?? null,
      height: image.height ?? null,
    }));

    const output = {
      listing_url: result?.metadata?.finalUrl ?? listingUrl,
      title: result?.metadata?.title ?? null,
      fields: result?.json?.success ? result.json.data : null,
      extraction_status: result?.json?.success ? "success" : "failed",
      extraction_error: result?.json?.success
        ? null
        : result?.json?.message ?? result?.json?.error_code ?? "No JSON extraction returned",
      floorplan_detection: floorplanCandidates.length ? "candidate_found" : "no_label_match",
      floorplan_candidates: floorplanCandidates,
      gallery_images: images.map((image) => ({
        url: image.url,
        alt: image.alt ?? null,
        width: image.width ?? null,
        height: image.height ?? null,
      })),
      gallery_status: result?.images?.success ? "success" : "failed",
      gallery_error: result?.images?.success
        ? null
        : result?.images?.message ?? result?.images?.error_code ?? null,
      credits_used: result?.key_metadata?.credits_consumed ?? null,
      credits_remaining: result?.key_metadata?.credits_remaining ?? null,
      request_id: result?.request_id ?? null,
    };

    console.log(JSON.stringify(output, null, 2));
    if (!response.headers.get("x-credits-used") && output.credits_used === null) {
      console.error("Credit usage was not included in the response.");
    }
  } finally {
    clearTimeout(timeout);
  }
}

main().catch((error) => {
  if (error.name === "AbortError") {
    console.error("Context.dev request timed out after 90 seconds.");
  } else {
    console.error(error.message);
  }
  process.exitCode = 1;
});
