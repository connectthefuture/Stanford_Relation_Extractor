package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.kbp.common.Maybe;
import edu.stanford.nlp.util.Triple;
import edu.stanford.nlp.util.logging.PrettyLoggable;
import edu.stanford.nlp.util.logging.Redwood;

import java.text.DecimalFormat;
import java.util.Arrays;

import static edu.stanford.nlp.util.logging.Redwood.Util.endTrack;
import static edu.stanford.nlp.util.logging.Redwood.Util.startTrack;

/**
 * Not to be confused with SFScore (the official scorer).
 * This class encapsulates statistics from evaluating the KBP system,
 * including importantly, precision, recall, and area under the curve.
 *
 * @author Gabor Angeli
 */
public class KBPScore implements PrettyLoggable {
  public final double precision;
  public final double recall;
  public final double f1;
  private final double[] precisions;
  private final double[] recalls;
  private final double accuracy;

  public KBPScore(double precision, double recall, double accuracy, double[] precisions, double[] recalls) {
    this.precision = precision;
    this.recall = recall;
    this.accuracy = accuracy;
    this.f1 = 2.0 * precision * recall / (precision + recall);
    this.precisions = precisions;
    this.recalls = recalls;
    if (precisions.length != recalls.length) {
      throw new IllegalArgumentException("Lengths of precisions and recalls must match");
    }
  }

  /**
   * The area under the precision recall curve,
   * @return An area, between 0.0 (worst) to 1.0 (best)
   */
  public double areaUnderPRCurve() {
    if (recalls.length == 0) return Double.NaN;
    final double firstRecall = recalls[0];
    double prevPrecision = precisions[0];
    double prevRecall = recalls[0];
    double sum = 0.0;
    for (int i=1; i<precisions.length; ++i) {
      final double recall = recalls[i];
      final double precision = precisions[i];
      sum += (recall - prevRecall) * (precision + prevPrecision) / 2.0;
      prevRecall = recall;
      prevPrecision = precision;
    }
    return sum / (1.0 - firstRecall);
  }

  /**
   * Return the optimal score (precision, recall, f1), tuned for F1.
   * @return A triple (precision, recall, f1)
   */
  public Triple<Double,Double,Double> optimalPrecisionRecallF1() {
    double max = Double.NEGATIVE_INFINITY;
    double p = Double.NaN;
    double r = Double.NaN;
    for (int i=0; i<precisions.length; ++i) {
      double candidate = f1(precisions[i], recalls[i]);
      if (!Double.isInfinite(candidate) && !Double.isNaN(candidate)){
        if (candidate > max) {
          max = candidate;
          p = precisions[i];
          r = recalls[i];
        }
      }
    }
    return Triple.makeTriple(p, r, max);
  }

  /**
   * Return the F1 score for the optimal point on the PR Curve
   * @return The optimal F1 value
   */
  public double optimalF1() {
    return optimalPrecisionRecallF1().third;
  }

  /**
   * Report an official score if available.
   */
  public KBPScore merge(Maybe<KBPScore> officialScore) {
    if (officialScore.isDefined()) {
      return new KBPScore(officialScore.get().precision, officialScore.get().recall, this.accuracy, this.precisions, this.recalls);
    } else {
      return this;
    }
  }

  @Override
  public void prettyLog(Redwood.RedwoodChannels channels, String description) {
    DecimalFormat df = new DecimalFormat("00.000");
    Triple<Double, Double, Double> optimal = optimalPrecisionRecallF1();
    startTrack(description);
    channels.log("|           Precision: " + df.format(precision * 100));
    channels.log("|              Recall: " + df.format(recall * 100));
    channels.log("|                  F1: " + df.format(f1(precision, recall) * 100));
    channels.log("|");
    channels.log("|   Optimal Precision: " + df.format(optimal.first * 100));
    channels.log("|      Optimal Recall: " + df.format(optimal.second * 100));
    channels.log("|          Optimal F1: " + df.format(optimal.third * 100));
    channels.log("|");
    channels.log("| Area Under PR Curve: " + areaUnderPRCurve());
    endTrack(description);
  }

  @Override
  public String toString() {
    return "[F1: " + new DecimalFormat("0.000").format(f1) + "] P " + precision + "; R " + recall + "; AUC " + areaUnderPRCurve();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof KBPScore)) return false;
    KBPScore kbpScore = (KBPScore) o;
    return Double.compare(kbpScore.precision, precision) == 0 && Double.compare(kbpScore.recall, recall) == 0 && Arrays.equals(precisions, kbpScore.precisions) && Arrays.equals(recalls, kbpScore.recalls);

  }

  @Override
  public int hashCode() {
    int result;
    long temp;
    temp = precision != +0.0d ? Double.doubleToLongBits(precision) : 0L;
    result = (int) (temp ^ (temp >>> 32));
    temp = recall != +0.0d ? Double.doubleToLongBits(recall) : 0L;
    result = 31 * result + (int) (temp ^ (temp >>> 32));
    return result;
  }

  public static double f1(double precision, double recall) {
    return 2.0 * precision * recall / (precision + recall);
  }
}
