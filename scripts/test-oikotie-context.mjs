#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DEFAULT_URL =
  "https://asunnot.oikotie.fi/myytavat-asunnot/helsinki/24784966";
const API_URL = "https://api.context.dev/v1/web/scrape";
const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function readOptions(name) {
  const values = [];
  for (let index = 0; index < process.argv.length; index += 1) {
    if (process.argv[index] === name && process.argv[index + 1]) {
      values.push(process.argv[index + 1]);
      index += 1;
    }
  }
  return values;
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

function isOikotiePropertyImage(image) {
  try {
    const url = new URL(image.url);
    return (
      url.hostname === "cdn.asunnot.oikotie.fi" &&
      url.pathname.includes("/ot-real-estate-mediabank-prod/") &&
      /\/source\/\d+/.test(url.pathname)
    );
  } catch {
    return false;
  }
}

function deduplicatePropertyImages(images) {
  const bySourceId = new Map();
  for (const image of images) {
    const sourceId = getSourceId(image) ?? image.url;
    const current = bySourceId.get(sourceId);
    const pixelArea = (image.width ?? 0) * (image.height ?? 0);
    const currentPixelArea = current ? (current.width ?? 0) * (current.height ?? 0) : -1;
    if (!current || pixelArea > currentPixelArea) bySourceId.set(sourceId, image);
  }
  return [...bySourceId.values()];
}

function getSourceId(image) {
  const match = image.url.match(/\/source\/(\d+)/);
  return match ? Number(match[1]) : null;
}

function getAlbumSegment(image) {
  try {
    const mediaPath = new URL(image.url).pathname.split("/ot-real-estate-mediabank-prod/")[1];
    const segments = mediaPath?.split("/");
    return segments?.length >= 4 ? segments[1] : null;
  } catch {
    return null;
  }
}

function narrowToLeadImageGroup(images) {
  const leadAlbum = images.length ? getAlbumSegment(images[0]) : null;
  if (leadAlbum === null) return { images, method: "all_large_media_no_album_key" };

  const group = images.filter((image) => getAlbumSegment(image) === leadAlbum);
  if (group.length > 1 && group.length < images.length) {
    return { images: group, method: "shared_album_segment_from_lead_image" };
  }
  return { images, method: "all_large_media_no_group" };
}

async function scrapeListing(listingUrl, apiKey) {
  const includeDebugImages = process.argv.includes("--debug-images");
  const openGallery = process.argv.includes("--open-gallery");
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
        ...(openGallery
          ? {
              sharedParams: {
                actions: [
                  {
                    type: "perform",
                    action:
                      "Click the Oikotie button labeled 'Katso kaikki kuvat' to open the full listing photo gallery.",
                  },
                  { type: "wait", milliseconds: 1500 },
                ],
              },
            }
          : {}),
        imageParams: { enrich: ["dimensions"] },
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

    const pageImages = result?.images?.success ? result.images.data ?? [] : [];
    const mediaBankImages = deduplicatePropertyImages(pageImages.filter(isOikotiePropertyImage));
    const largeMediaImages = mediaBankImages.filter(
      (image) => image.width >= 900 && image.height >= 600,
    );
    // Opening the full gallery is intended to expose every listing photo, including
    // floorplans. Keep all large media-bank images instead of grouping by URL path:
    // photos in one gallery can have different album segments.
    const mediaForGrouping = largeMediaImages.length ? largeMediaImages : mediaBankImages;
    const selectedImages = openGallery
      ? { images: mediaForGrouping, method: "all_images_after_opening_gallery" }
      : narrowToLeadImageGroup(mediaForGrouping);
    const propertyImages = selectedImages.images;
    const floorplanCandidates = propertyImages.filter(isFloorplanImage).map((image) => ({
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
      floorplan_detection: floorplanCandidates.length
        ? "candidate_found_from_image_label"
        : propertyImages.length
          ? "visual_review_needed"
          : "no_property_images_found",
      floorplan_candidates: floorplanCandidates,
      page_image_count: pageImages.length,
      media_bank_image_count: mediaBankImages.length,
      property_image_candidate_count: propertyImages.length,
      property_image_selection: selectedImages.method,
      property_image_candidates: propertyImages.map((image) => ({
        url: image.url,
        alt: image.alt ?? null,
        width: image.width ?? null,
        height: image.height ?? null,
      })),
      ...(includeDebugImages
        ? {
            debug_media_images: mediaBankImages.map((image) => ({
              url: image.url,
              alt: image.alt ?? null,
              width: image.width ?? null,
              height: image.height ?? null,
              source_id: getSourceId(image),
              album_segment: getAlbumSegment(image),
            })),
          }
        : {}),
      gallery_status: result?.images?.success ? "success" : "failed",
      gallery_error: result?.images?.success
        ? null
        : result?.images?.message ?? result?.images?.error_code ?? null,
      credits_used: result?.key_metadata?.credits_consumed ?? null,
      credits_remaining: result?.key_metadata?.credits_remaining ?? null,
      request_id: result?.request_id ?? null,
    };

    return output;
  } finally {
    clearTimeout(timeout);
  }
}

async function main() {
  const requestedUrls = readOptions("--url");
  const listingUrls = (requestedUrls.length ? requestedUrls : [DEFAULT_URL]).map(validateListingUrl);
  const apiKey = await loadApiKey();

  if (!apiKey) {
    console.error("Missing CONTEXT_DEV_API_KEY. Add it to the project .env file first.");
    process.exitCode = 1;
    return;
  }

  const results = [];
  for (const listingUrl of listingUrls) {
    try {
      results.push(await scrapeListing(listingUrl, apiKey));
    } catch (error) {
      results.push({
        listing_url: listingUrl,
        extraction_status: "request_failed",
        error: error.name === "AbortError" ? "Timed out after 90 seconds" : error.message,
      });
    }
  }

  console.log(JSON.stringify({ results }, null, 2));
}

main().catch((error) => {
  if (error.name === "AbortError") {
    console.error("Context.dev request timed out after 90 seconds.");
  } else {
    console.error(error.message);
  }
  process.exitCode = 1;
});
