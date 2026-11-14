import { createClient } from "@supabase/supabase-js";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(request: Request, { params }: { params: Promise<{ planId: string }> }) {
  const { planId } = await params;
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!supabaseUrl || !anonKey) {
    return NextResponse.json({ detail: "Supabase is not configured on the server." }, { status: 503 });
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return NextResponse.json({ detail: "Upload request was not valid." }, { status: 400 });
  }
  const file = form.get("file");
  const filename = form.get("filename");
  if (!(file instanceof File) || typeof filename !== "string" || !["application/pdf", "image/png", "image/jpeg", "image/webp"].includes(file.type)) {
    return NextResponse.json({ detail: "Please upload a PDF or an image file." }, { status: 400 });
  }

  const bucket = process.env.NEXT_PUBLIC_SUPABASE_BUCKET || "plans";
  const safeName = filename.replace(/[^a-zA-Z0-9._-]/g, "-");
  const path = `${planId}/${crypto.randomUUID()}-${safeName}`;
  const supabase = createClient(supabaseUrl, anonKey);
  const { error: recordError } = await supabase.rpc("create_plan_with_image", {
    p_plan_id: planId,
    p_bucket_id: bucket,
    p_object_path: path,
    p_role: "source",
    p_metadata: { filename, content_type: file.type },
  });
  if (recordError) {
    return NextResponse.json({ detail: `Could not create plan: ${recordError.message}` }, { status: 502 });
  }

  const { error: uploadError } = await supabase.storage.from(bucket).upload(path, file, {
    contentType: file.type,
    upsert: false,
  });
  if (uploadError) {
    await supabase.rpc("set_plan_image_upload_status", { p_plan_id: planId, p_object_path: path, p_status: "failed" });
    return NextResponse.json({ detail: `Upload failed: ${uploadError.message}` }, { status: 502 });
  }

  const { error: statusError } = await supabase.rpc("set_plan_image_upload_status", {
    p_plan_id: planId,
    p_object_path: path,
    p_status: "uploaded",
  });
  if (statusError) {
    return NextResponse.json({ detail: "Image uploaded, but its saved status could not be confirmed." }, { status: 502 });
  }
  return NextResponse.json({ planId });
}
