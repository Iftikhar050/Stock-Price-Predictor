// Turns the raw 4-model ensemble prediction into a compact Bullish/Neutral/Bearish
// "signal" - the thing none of the reference competitor sites expose.
const NEUTRAL_BAND_PCT = 0.3; // avg predicted move smaller than this counts as flat

export function computeSignal(prediction) {
  if (!prediction) return null;
  const { current_price, rf_predicted_price, lr_predicted_price, xgb_predicted_price, lstm_predicted_price, model_agreement_score } = prediction;
  const preds = [rf_predicted_price, lr_predicted_price, xgb_predicted_price, lstm_predicted_price].filter(
    (p) => typeof p === "number" && !Number.isNaN(p)
  );
  if (!preds.length || typeof current_price !== "number" || !current_price) return null;

  const avgPredicted = preds.reduce((sum, p) => sum + p, 0) / preds.length;
  const pctChange = ((avgPredicted - current_price) / current_price) * 100;

  let direction = "neutral";
  if (pctChange > NEUTRAL_BAND_PCT) direction = "bullish";
  else if (pctChange < -NEUTRAL_BAND_PCT) direction = "bearish";

  return {
    direction,
    pctChange,
    confidence: typeof model_agreement_score === "number" ? model_agreement_score : null,
  };
}
