package edu.stanford.nlp.kbp.slotfilling.process;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.common.CollectionUtils;
import edu.stanford.nlp.kbp.common.KBPSlotFill;
import edu.stanford.nlp.kbp.common.NERTag;
import edu.stanford.nlp.kbp.common.Props;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.semgrex.SemgrexMatcher;
import edu.stanford.nlp.semgraph.semgrex.SemgrexPattern;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.StringUtils;

import java.util.*;

/**
 * A collection of features which can be used for relation classification.
 *
 * <p>
 *   TO IMPLEMENT YOUR OWN FEATURE:
 * </p>
 * <ol>
 *   <li>Create a class extending {@link edu.stanford.nlp.kbp.slotfilling.process.FeatureProvider} here.</li>
 *   <li>Register this class in the {@link edu.stanford.nlp.kbp.slotfilling.process.Feature} enum.</li>
 *   <li>IMPORTANT: Unit test your class in FeatureProvidersTest.</li>
 * </ol>
 *
 * @see edu.stanford.nlp.kbp.slotfilling.process.FeatureProvider
 *
 * @author Gabor Angeli
 */
@SuppressWarnings("unchecked")
public class FeatureProviders {
  /** Hide the constructor -- static items only */
  private FeatureProviders() { }

  /**
   * An abstract class for extracting features in the lexical span between the entity and
   * slot mention.
   */
  private static abstract class SpanBigramFeature extends FeatureProvider {
    public SpanBigramFeature(String prefix) {
      super(prefix);
    }

    /** {@inheritDoc} */
    @Override
    protected Collection<String> featureValues(Featurizable factory) {
      Span subj = factory.subj;
      Span obj = factory.obj;
      List<CoreLabel> tokens = factory.tokens;
      if (subj.end() < obj.start() || obj.end() < subj.start()) {
        int betweenSpanStart = Math.min(subj.end(), obj.end());
        int betweenSpanEnd = Math.max(subj.start(), obj.start());
        Collection<String> values = new HashSet<>(betweenSpanEnd - betweenSpanStart);

        // Span Bag-of-words
        for (int i = betweenSpanStart; i < betweenSpanEnd; ++i) {
          // (get word)
          if (i < tokens.size() && i >= 0) {
            String word = tokens.get(i).word();
            String prevWord = "^";
            if (i > betweenSpanStart && i > 0) {
              prevWord = tokens.get(i - 1).word();
            }
            // (get lemma)
            String lemma = tokens.get(i).lemma();
            String prevLemma = "^";
            if (i > betweenSpanStart && i > 0) { prevLemma = tokens.get(i - 1).lemma(); }
            // (get NER)
            String ner = tokens.get(i).ner();
            String prevNER = "^";
            if (i > betweenSpanStart && i > 0) { prevNER = tokens.get(i - 1).ner(); }
            // (register feature)
            String feat = featureValueOrNull(prevWord, word, prevLemma, lemma, prevNER, ner);
            if (feat != null) {
              values.add(feat);
            }
          }
        }
        // (register final feature)
        CoreLabel lastToken = tokens.get(betweenSpanEnd - 1);
        String finalFeat = finalFeatureValueOrNull(lastToken.word(), lastToken.lemma(), lastToken.ner());
        if (finalFeat != null) {
          values.add(finalFeat);
        }
        return values;
      } else {
        return Collections.EMPTY_LIST;
      }
    }

    /** Extract a particular type of feature, given a bunch of data for a given token in the lexical span */
    protected abstract String featureValueOrNull(String prevWord, String word, String prevLemma, String lemma, String prevNER, String ner);

    /** An optinal function for the last token in the span */
    protected String finalFeatureValueOrNull(String word, String lemma, String ner) {
      return null;
    }
  }

  /**
   * An abstract class for extracting features along the dependency path from the entity to the slot value.
   */
  private static abstract class DepPathFeature extends FeatureProvider {
    public DepPathFeature(String prefix) {
      super(prefix);
    }

    @Override
    protected Collection<String> featureValues(Featurizable factory) {
      // Get dependency nodes
      IndexedWord subjNode = null;
      for (int i = factory.subj.start(); i < factory.subj.end(); ++i) {
        if (subjNode == null) {
          subjNode = factory.dependencies.getNodeByIndexSafe(factory.subj.start() + 1);
        }
      }
      IndexedWord objNode = null;
      for (int i = factory.obj.start(); i < factory.obj.end(); ++i) {
        objNode = factory.dependencies.getNodeByIndexSafe(factory.obj.start() + 1);
      }
      if (subjNode == null || objNode == null) {
        return Collections.EMPTY_LIST;
      }
      // Populate features
      Collection<String> features = new HashSet<>();
      List<IndexedWord> path = factory.dependencies.getShortestUndirectedPathNodes(subjNode, objNode);
      if (path != null) {
        for (IndexedWord word : path) {
          if (((word.index() > factory.subj.end() || word.index() <= factory.subj.start())) &&
              ((word.index() > factory.obj.end() || word.index() <= factory.obj.start()))) {
            String feat = featureValueOrNull(word.word(), word.lemma(), word.ner());
            if (feat != null) {
              features.add(feat);
            }
          }
        }
      }
      return features;
    }

    /** Extract a particular feature for a given node along the path */
    protected abstract String featureValueOrNull(String word, String lemma, String ner);
  }

  /**
   * An abstract class for extracting a feature pertaining to the NER signature of the relation mention.
   */
  private static abstract class NERSignatureFeature extends FeatureProvider {
    public NERSignatureFeature(String prefix) {
      super(prefix);
    }

    /** {@inheritDoc} */
    @Override
    protected Collection<String> featureValues(Featurizable factory) {
      // Types
      // (subject)
      Counter<String> subjNERCandidates = new ClassicCounter<>();
      for (int i : factory.subj) {
        if (i >= 0 && i < factory.tokens.size() && !factory.tokens.get(i).ner().equals(Props.NER_BLANK_STRING)) { subjNERCandidates.incrementCount(factory.tokens.get(i).ner()); }
      }
      String subjNER = subjNERCandidates.size() > 0 ? Counters.argmax(subjNERCandidates) : NERTag.MISC.name;
      // (object)
      Counter<String> objNERCandidates = new ClassicCounter<>();
      for (int i : factory.obj) {
        if (i >= 0 && i < factory.tokens.size() && !factory.tokens.get(i).ner().equals(Props.NER_BLANK_STRING)) { objNERCandidates.incrementCount(factory.tokens.get(i).ner()); }
      }
      String objNER = objNERCandidates.size() > 0 ? Counters.argmax(objNERCandidates) : NERTag.MISC.name;
      String feat = featureValueOrNull(subjNER, objNER);
      if (feat != null) {
        return Collections.singletonList(feat);
      } else {
        return Collections.EMPTY_LIST;
      }
    }

    /** Extract a particular feature, given the entity and slot value NER */
    protected abstract String featureValueOrNull(String subjNER, String objNER);
  }

  /** Word unigram in the lexical span between the entity and slot value */
  public static class LexBetweenWordUnigram extends SpanBigramFeature {
    public LexBetweenWordUnigram(String prefix) {
      super(prefix);
    }

    @Override
    protected String featureValueOrNull(String prevWord, String word, String prevLemma, String lemma, String prevNER, String ner) {
      return word;
    }
  }

  /** Lemma unigram in the lexical span between the entity and slot value */
  public static class LexBetweenLemmaUnigram extends SpanBigramFeature {
    public LexBetweenLemmaUnigram(String prefix) {
      super(prefix);
    }

    @Override
    protected String featureValueOrNull(String prevWord, String word, String prevLemma, String lemma, String prevNER, String ner) {
      return lemma;
    }
  }

  /** Word bigram in the lexical span between the entity and slot value */
  public static class LexBetweenWordBigram extends SpanBigramFeature {
    public LexBetweenWordBigram(String prefix) {
      super(prefix);
    }

    @Override
    protected String featureValueOrNull(String prevWord, String word, String prevLemma, String lemma, String prevNER, String ner) {
      return prevWord + "_" + word;
    }

    @Override
    protected String finalFeatureValueOrNull(String word, String lemma, String ner) {
      return word + "_$";
    }
  }

  /** Lemma bigram in the lexical span between the entity and slot value */
  public static class LexBetweenLemmaBigram extends SpanBigramFeature {
    public LexBetweenLemmaBigram(String prefix) {
      super(prefix);
    }

    @Override
    protected String featureValueOrNull(String prevWord, String word, String prevLemma, String lemma, String prevNER, String ner) {
      return prevLemma + "_" + lemma;
    }

    @Override
    protected String finalFeatureValueOrNull(String word, String lemma, String ner) {
      return lemma + "_$";
    }
  }

  /** An indicator for if we cross another NER tag in the lexical span between the entity and slot value */
  public static class LexBetweenNER extends SpanBigramFeature {
    public LexBetweenNER(String prefix) {
      super(prefix);
    }

    @Override
    protected String featureValueOrNull(String prevWord, String word, String prevLemma, String lemma, String prevNER, String ner) {
      if (!ner.equals(Props.NER_BLANK_STRING)) {
        return ner;
      } else {
        return null;
      }
    }
  }

  /** A feature for the entity type */
  public static class NERSignatureEntity extends NERSignatureFeature {
    public NERSignatureEntity(String prefix) {
      super(prefix);
    }

    @Override
    protected String featureValueOrNull(String subjNER, String objNER) {
      return subjNER;
    }
  }

  /** A feature for the slot value type */
  public static class NERSignatureSlotValue extends NERSignatureFeature {
    public NERSignatureSlotValue(String prefix) {
      super(prefix);
    }

    @Override
    protected String featureValueOrNull(String subjNER, String objNER) {
      return objNER;
    }
  }

  /** A feature for the joint NER type of the entity and slot value */
  public static class NERSignature extends NERSignatureFeature {
    public NERSignature(String prefix) {
      super(prefix);
    }

    @Override
    protected String featureValueOrNull(String subjNER, String objNER) {
      return subjNER + "_" + objNER;
    }
  }

  /** A feature for whether there is punctuation between the entity and slot value */
  public static class LexBetweenPunctuation extends FeatureProvider {
    public LexBetweenPunctuation(String prefix) {
      super(prefix);
    }

    @Override
    protected Collection<String> featureValues(Featurizable factory) {
      boolean isSplitByPunctuation = false;
      int betweenSpanStart = Math.min(factory.subj.end(), factory.obj.end());
      int betweenSpanEnd = Math.max(factory.subj.start(), factory.obj.start());
      for (int i = betweenSpanStart; i < betweenSpanEnd; ++i) {
        if (i >= 0 && i < factory.tokens.size() && factory.tokens.get(i).originalText() != null && factory.tokens.get(i).originalText().length() == 1 &&
            "()[]{}<>.,;!?\"'-".contains(factory.tokens.get(i).originalText().substring(0, 1))) {
          isSplitByPunctuation = true;
        }
      }
      if (isSplitByPunctuation) {
        return Collections.singletonList("");
      } else {
        return Collections.EMPTY_LIST;
      }
    }
  }

  /** The word unigram along the dependency path between the entity and slot value */
  public static class DepBetweenWordUnigram extends DepPathFeature {
    public DepBetweenWordUnigram(String prefix) {
      super(prefix);
    }

    @Override
    protected String featureValueOrNull(String word, String lemma, String ner) {
      return word;
    }
  }

  /** The lemma unigram along the dependency path between the entity and slot value */
  public static class DepBetweenLemmaUnigram extends DepPathFeature {
    public DepBetweenLemmaUnigram(String prefix) {
      super(prefix);
    }

    @Override
    protected String featureValueOrNull(String word, String lemma, String ner) {
      return lemma;
    }
  }

  /** The NER tags of entries along the dependency path between the entity and slot value */
  public static class DepBetweenNER extends DepPathFeature {
    public DepBetweenNER(String prefix) {
      super(prefix);
    }

    @Override
    protected String featureValueOrNull(String word, String lemma, String ner) {
      if (!ner.equals(Props.NER_BLANK_STRING)) {
        return ner;
      } else {
        return null;
      }
    }
  }

  /** A class capturing some very simple dependency relations */
  public static class OpenIESimplePatterns extends FeatureProvider {
    public final SemgrexPattern subj_obj;

    public OpenIESimplePatterns(String prefix) {
      super(prefix);

      String NER = "ner:/" + StringUtils.join(CollectionUtils.map(NERTag.values(), tag -> tag.name), "|") + "/";

      // Relation: verb
      subj_obj = SemgrexPattern
          .compile("{tag:/VB.*/;ner:O}=rel >/.?subj.*/ {" + NER + "}=subj >>/.?obj.*/ {" + NER + "}=obj");
    }

    @Override
    protected Collection<String> featureValues(Featurizable factory) {
      if (factory.dependencies.getRoots().isEmpty()) { return Collections.EMPTY_LIST; }
      List<String> features = new ArrayList<>();
      SemgrexMatcher matcher = subj_obj.matcher(factory.dependencies);
      if (matcher.find()) {
        IndexedWord subj = matcher.getNode("subj");
        IndexedWord obj = matcher.getNode("obj");
        if (factory.subj.contains(subj.index() - 1) &&
            factory.obj.contains(obj.index() - 1)) {
          features.add("subj<-" + matcher.getNode("rel").originalText() + "->obj");
        } else if (factory.subj.contains(obj.index() - 1) &&
            factory.obj.contains(subj.index() - 1)) {
          features.add("obj<-" + matcher.getNode("rel").originalText() + "->subj");
        }
      }
      return features;
    }
  }

  public static class OpenIERelation extends FeatureProvider {
    public OpenIERelation(String prefix) {
      super(prefix);
    }

    @Override
    protected Collection<String> featureValues(Featurizable factory) {
      if (factory.openIE.isEmpty()) { return Collections.EMPTY_LIST; }
      List<String> features = new ArrayList<>();
      for (KBPSlotFill openieFill : factory.openIE) {
        if (openieFill.provenance.isDefined()) {
          for (Span entitySpan : openieFill.provenance.get().entityMentionInSentence) {
            for (Span slotSpan : openieFill.provenance.get().slotValueMentionInSentence) {
              if (Span.overlaps(entitySpan, factory.subj) && Span.overlaps(slotSpan, factory.obj)) {
                features.add("subj<-" + openieFill.key.relationName + "->obj");
              } else if (Span.overlaps(entitySpan, factory.obj) && Span.overlaps(slotSpan, factory.subj)) {
                features.add("obj<-" + openieFill.key.relationName + "->subj");

              }
            }
          }
        } else {
          features.add("subj<-" + openieFill.key.relationName + "->obj");
        }
      }
      return features;
    }
  }


}
