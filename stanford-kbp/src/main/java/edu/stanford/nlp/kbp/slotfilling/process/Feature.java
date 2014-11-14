package edu.stanford.nlp.kbp.slotfilling.process;

import edu.stanford.nlp.util.StringUtils;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import static edu.stanford.nlp.kbp.slotfilling.process.FeatureProviders.*;

/**
 * An enumeration of possible feature types.
 *
 * <p>
 *   TO IMPLEMENT YOUR OWN [NEW] FEATURE:
 * </p>
 * <ol>
 *   <li>Create a class extending {@link edu.stanford.nlp.kbp.slotfilling.process.FeatureProvider}
 *       in {@link edu.stanford.nlp.kbp.slotfilling.process.FeatureProviders}.</li>
 *   <li>Register this class here.</li>
 *   <li>IMPORTANT: Unit test your class in FeatureProvidersTest.</li>
 * </ol>
 *
 * <p>
 *   TO IMPLEMENT A JOINT FEATURE:
 *   simply link a bunch of existing feature extractors in the constructor.
 * </p>
 *
 * @author Gabor Angeli
 */
public enum Feature {
  // Lexical features
  LEX_BETWEEN_WORD_UNIGRAM(  new LexBetweenWordUnigram("lex_between_word_unigram")),
  LEX_BETWEEN_WORD_BIGRAM(   new LexBetweenWordBigram("lex_between_word_bigram")),
  LEX_BETWEEN_LEMMA_UNIGRAM( new LexBetweenLemmaUnigram("lex_between_lemma_unigram")),
  LEX_BETWEEN_LEMMA_BIGRAM(  new LexBetweenLemmaBigram("lex_between_lemma_bigram")),
  LEX_BETWEEN_NER(           new LexBetweenNER("lex_between_ner")),
  LEX_BETWEEN_PUNCTUATION(   new LexBetweenPunctuation("lex_split_by_punctuation")),

  // Dependency features
  DEP_BETWEEN_WORD_UNIGRAM(  new DepBetweenWordUnigram("dep_between_word_unigram")),
  DEP_BETWEEN_LEMMA_UNIGRAM( new DepBetweenLemmaUnigram("dep_between_lemma_unigram")),
  DEP_BETWEEN_NER(           new DepBetweenNER("dep_between_ner")),

  // NER Signature
  NER_SIGNATURE_ENTITY(      new NERSignatureEntity("ner_signature_entity")),
  NER_SIGNATURE_SLOT_VALUE(  new NERSignatureSlotValue("ner_signature_slot_value")),
  NER_SIGNATURE(             new NERSignature("ner_signature")),

  // Type-tagged OpenIE patterns
  OPENIE_SIMPLE_PATTERNS(  new OpenIESimplePatterns("openie_simple_pattern"), new NERSignature("ner_signature")),
  OPENIE_RELATIONS(        new OpenIERelation("openie_relation"), new NERSignature("ner_signature")),

  // Type-tagged unigram features
  LEX_BETWEEN_LEMMA_UNIGRAM__NER_SIGNATURE( new LexBetweenLemmaUnigram("lex_between_lemma_unigram"), new NERSignature("ner_signature")),
  DEP_BETWEEN_LEMMA_UNIGRAM__NER_SIGNATURE( new LexBetweenLemmaUnigram("dep_between_lemma_unigram"), new NERSignature("ner_signature")),
  ;

  static {
    Set<String> featureNames = new HashSet<>();
    for (Feature feat : Feature.values()) {
      if (featureNames.contains(feat.provider.prefix)) { throw new IllegalStateException("Duplicate feature name: " + feat.provider.prefix); }
      featureNames.add(feat.provider.prefix);
    }
  }

  public final FeatureProvider provider;
  public final boolean isConjoinedFeature;

  Feature(FeatureProvider... providers) {
    if (providers.length == 1) {
      this.provider = providers[0];
      this.isConjoinedFeature = false;
    } else {
      this.provider = join(providers);
      this.isConjoinedFeature = true;
    }
  }

  public static Feature fromString(String str) {
    str = str.toLowerCase();
    for (Feature feat : Feature.values()) {
      if (feat.name().toLowerCase().equals(str)) { return feat; }
      if (feat.provider.prefix.toLowerCase().equals(str)) { return feat; }
      if (feat.provider.getClass().getSimpleName().toLowerCase().equals(str)) { return feat; }
    }
    throw new IllegalArgumentException("No such feature: " + str);
  }

  private static FeatureProvider join(final FeatureProvider... providers) {
    if (providers.length == 0) {
      throw new IllegalArgumentException("At least one provider must be available");
    }
    final String[] prefixes = new String[providers.length];
    for (int i = 0; i < prefixes.length; ++i) {
      prefixes[i] = providers[i].prefix;
    }
    return new FeatureProvider(StringUtils.join(prefixes, "__")) {
      @Override
      protected Collection<String> featureValues(Featurizable factory) {
        Collection<String> fringe = providers[0].featureValues(factory);
        for (int i = 1; i < providers.length; ++i) {
          Collection<String> newFringe = new HashSet<>();
          Collection<String> newFeatures = providers[i].featureValues(factory);
          for (String existingFeat : fringe) {
            for (String newFeat : newFeatures) {
              newFringe.add(existingFeat + "+" + newFeat);
            }
          }
          fringe = newFringe;
        }
        return fringe;
      }
    };
  }
}
