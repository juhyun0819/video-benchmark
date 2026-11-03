const BASE = "https://huggingface.co/datasets/mmm8383/NextMe-800/resolve/main";

export const HF_ACCESS_DENIED = "HF 접근 거부: NextMe-800 접근 승인과 HF_TOKEN을 확인하세요";

/**
 * Returns a function that downloads a text file from the gated NextMe-800
 * dataset repo. The token only ever goes into the Authorization header; it is
 * never included in thrown messages.
 */
export function makeHfFetcher(
  token: string,
  fetchImpl: typeof fetch = fetch,
): (path: string) => Promise<string> {
  return async (path: string) => {
    const res = await fetchImpl(`${BASE}/${path}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (res.status === 401 || res.status === 403) throw new Error(HF_ACCESS_DENIED);
    if (!res.ok) throw new Error(`HF 요청 실패: HTTP ${res.status} ${path}`);
    return res.text();
  };
}
