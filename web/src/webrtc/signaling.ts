type OfferRequest = {
  sdp: string;
  type: RTCSdpType;
  session_id?: string;
};

type OfferResponse = {
  sdp: string;
  type: RTCSdpType;
  session_id: string;
};

type IceRequest = {
  session_id: string;
  candidate: string;
  sdpMid?: string | null;
  sdpMLineIndex?: number | null;
};

const configuredBaseUrl = import.meta.env.VITE_SIGNALING_BASE_URL;
const baseUrl = (configuredBaseUrl && configuredBaseUrl.trim().length > 0 ? configuredBaseUrl : "/api").replace(/\/$/, "");

export function getSignalingBaseUrl(): string {
  return baseUrl;
}

export async function sendOffer(payload: OfferRequest): Promise<OfferResponse> {
  const response = await fetch(`${baseUrl}/webrtc/offer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(`Offer failed with status ${response.status}`);
  }

  return (await response.json()) as OfferResponse;
}

export async function sendIceCandidate(payload: IceRequest): Promise<void> {
  const response = await fetch(`${baseUrl}/webrtc/ice`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(`ICE failed with status ${response.status}`);
  }
}
