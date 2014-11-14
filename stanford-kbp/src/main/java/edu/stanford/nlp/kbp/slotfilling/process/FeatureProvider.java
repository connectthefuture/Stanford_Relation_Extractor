package edu.stanford.nlp.kbp.slotfilling.process;

import edu.stanford.nlp.stats.Counter;

import java.util.Collection;

/**
 * An incarnation of a feature provider.
 * An instance of this class should provide one type of feature, taking as input a
 * {@link edu.stanford.nlp.kbp.slotfilling.process.Featurizable} -- an abstract incarnation of a
 * relation mention to be featurized.
 *
 * @see edu.stanford.nlp.kbp.slotfilling.process.FeatureProviders
 *
 * @author Gabor Angeli
 */
public abstract class FeatureProvider {
  public final String prefix;

  public FeatureProvider(String prefix){ this.prefix = prefix; }

  public final void apply(Featurizable factory, Counter<String> features) {
    assert features != null;
    for (String value : featureValues(factory)) {
      features.incrementCount(prefix + "_" + value);
    }
  }

  protected abstract Collection<String> featureValues(Featurizable factory);
}
