package edu.stanford.nlp.kbp.slotfilling.ir.query;

import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.similarities.*;

import java.util.LinkedList;
import java.util.List;

/**
 * Different dimensions along which a Lucene query can be parameterized.
 *
 * Adding a parameter to this class should be registered in a number of places:
 *   - The constructor
 *   - The equals() and hashCode() methods
 *   - A method to set that parameter
 *   - The all() method (returning all possible parameter configurations)
 *
 * @author Gabor Angeli
 * @author Angel Chang
 */
public class LuceneQuerierParams {

  public static enum PhraseSemantics {
    PHRASE,
    SPAN_ORDERED,
    SPAN_UNORDERED,
    UNIGRAMS_MUST,
    UNIGRAMS_SHOULD
  }

  public final Similarity similarity;
  public final PhraseSemantics phraseSemantics;
  public final Occur conjunctionMode;
  public final boolean fuzzy;
  public final boolean queryNERTag;
  public final boolean querySlotFill;
  public final boolean querySlotFillType;
  public final boolean queryRelation;
  public final boolean caseSensitive;
  public final boolean queryPreTokenizedText;  // queries text that is tokenized using Stanford tokenizer
  public final int slop;
  public final float entityBoost;
  public final float slotfillBoost;

  public static final int MAX_SLOP = 5;  // Max allowed slop to use when enumerating all possible query params

  public LuceneQuerierParams(Similarity similarity, PhraseSemantics phraseSemantics, Occur conjunctionMode,
                             boolean fuzzy, boolean queryNERTag,
                             boolean querySlotFill, boolean querySlotFillType, boolean queryRelation,
                             boolean caseSensitive, boolean queryPreTokenizedText, int slop,
                             float entityBoost, float slotfillBoost) {
    this.similarity = similarity;
    this.phraseSemantics = phraseSemantics;
    this.conjunctionMode = conjunctionMode;
    this.fuzzy = fuzzy;
    this.queryNERTag = queryNERTag;
    this.querySlotFill = querySlotFill;
    this.querySlotFillType = querySlotFillType;
    this.queryRelation = queryRelation;
    this.caseSensitive = caseSensitive;
    this.queryPreTokenizedText = queryPreTokenizedText;
    this.slop = slop;
    this.entityBoost = entityBoost;
    this.slotfillBoost = slotfillBoost;
  }

  public LuceneQuerierParams withSimilarity(Similarity x) {
    return new LuceneQuerierParams(x, phraseSemantics, conjunctionMode, fuzzy, queryNERTag, querySlotFill, querySlotFillType, queryRelation, caseSensitive, queryPreTokenizedText, slop, entityBoost, slotfillBoost);
  }

  public LuceneQuerierParams withPhraseSemantics(PhraseSemantics x) {
    return new LuceneQuerierParams(similarity, x, conjunctionMode, fuzzy, queryNERTag, querySlotFill, querySlotFillType, queryRelation, caseSensitive, queryPreTokenizedText, slop, entityBoost, slotfillBoost);
  }

  public LuceneQuerierParams withConjunctionMode(Occur x) {
    return new LuceneQuerierParams(similarity, phraseSemantics, x, fuzzy, queryNERTag, querySlotFill, querySlotFillType, queryRelation, caseSensitive, queryPreTokenizedText, slop, entityBoost, slotfillBoost);
  }

  public LuceneQuerierParams withFuzzy(boolean x) {
    return new LuceneQuerierParams(similarity, phraseSemantics, conjunctionMode, x, queryNERTag, querySlotFill, querySlotFillType, queryRelation, caseSensitive, queryPreTokenizedText, slop, entityBoost, slotfillBoost);
  }

  public LuceneQuerierParams withNERTag(boolean x) {
    return new LuceneQuerierParams(similarity, phraseSemantics, conjunctionMode, fuzzy, x, querySlotFill, querySlotFillType, queryRelation, caseSensitive, queryPreTokenizedText, slop, entityBoost, slotfillBoost);
  }

  public LuceneQuerierParams withSlotFill(boolean x) {
    return new LuceneQuerierParams(similarity, phraseSemantics, conjunctionMode, fuzzy, queryNERTag, x, querySlotFillType, queryRelation, caseSensitive, queryPreTokenizedText, slop, entityBoost, slotfillBoost);
  }

  public LuceneQuerierParams withSlotFillType(boolean x) {
    return new LuceneQuerierParams(similarity, phraseSemantics, conjunctionMode, fuzzy, queryNERTag, querySlotFill, x, queryRelation, caseSensitive, queryPreTokenizedText, slop, entityBoost, slotfillBoost);
  }

  public LuceneQuerierParams withRelation(boolean x) {
    return new LuceneQuerierParams(similarity, phraseSemantics, conjunctionMode, fuzzy, queryNERTag, querySlotFill, querySlotFillType, x, caseSensitive, queryPreTokenizedText, slop, entityBoost, slotfillBoost);
  }

  public LuceneQuerierParams withCaseSensitive(boolean x) {
    return new LuceneQuerierParams(similarity, phraseSemantics, conjunctionMode, fuzzy, queryNERTag, querySlotFill, querySlotFillType, queryRelation, x, queryPreTokenizedText, slop, entityBoost, slotfillBoost);
  }

  public LuceneQuerierParams withPreTokenizedText(boolean x) {
    return new LuceneQuerierParams(similarity, phraseSemantics, conjunctionMode, fuzzy, queryNERTag, querySlotFill, querySlotFillType, queryRelation, caseSensitive, x, slop, entityBoost, slotfillBoost);
  }

  public LuceneQuerierParams withSlop(int x) {
    return new LuceneQuerierParams(similarity, phraseSemantics, conjunctionMode, fuzzy, queryNERTag, querySlotFill, querySlotFillType, queryRelation, caseSensitive, queryPreTokenizedText, x, entityBoost, slotfillBoost);
  }

  @SuppressWarnings("UnusedDeclaration")
  public LuceneQuerierParams withEntityBoost(float x) {
    return new LuceneQuerierParams(similarity, phraseSemantics, conjunctionMode, fuzzy, queryNERTag, querySlotFill, querySlotFillType, queryRelation, caseSensitive, queryPreTokenizedText, slop, x, slotfillBoost);
  }

  @SuppressWarnings("UnusedDeclaration")
  public LuceneQuerierParams withSlotFillBoost(float x) {
    return new LuceneQuerierParams(similarity, phraseSemantics, conjunctionMode, fuzzy, queryNERTag, querySlotFill, querySlotFillType, queryRelation, caseSensitive, queryPreTokenizedText, slop, entityBoost, x);
  }

  // Does the slop value affect the query?
  public boolean slopIsRelevant() {
    // Currently slop is used for the span queries (can also be applied for phrase semantics as well...)
    return (PhraseSemantics.SPAN_ORDERED.equals(phraseSemantics) || PhraseSemantics.SPAN_UNORDERED.equals(phraseSemantics));
  }

  @SuppressWarnings("RedundantIfStatement")
  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof LuceneQuerierParams)) return false;

    LuceneQuerierParams that = (LuceneQuerierParams) o;

    if (fuzzy != that.fuzzy) return false;
    if (queryNERTag != that.queryNERTag) return false;
    if (queryRelation != that.queryRelation) return false;
    if (querySlotFill != that.querySlotFill) return false;
    if (querySlotFillType != that.querySlotFillType) return false;
    if (conjunctionMode != that.conjunctionMode) return false;
    if (phraseSemantics != that.phraseSemantics) return false;
    if (caseSensitive != that.caseSensitive) return false;
    if (queryPreTokenizedText != that.queryPreTokenizedText) return false;
    if (slop != that.slop) return false;
    if (!similarity.equals(that.similarity)) return false;
    if (entityBoost != that.entityBoost) return false;
    if (slotfillBoost != that.slotfillBoost) return false;

    return true;
  }

  @Override
  public int hashCode() {
    int result = similarity.hashCode();
    result = 31 * result + phraseSemantics.hashCode();
    result = 31 * result + conjunctionMode.hashCode();
    result = 31 * result + (fuzzy ? 1 : 0);
    result = 31 * result + (queryNERTag ? 1 : 0);
    result = 31 * result + (querySlotFill ? 1 : 0);
    result = 31 * result + (querySlotFillType ? 1 : 0);
    result = 31 * result + (queryRelation ? 1 : 0);
    result = 31 * result + (caseSensitive ? 1 : 0);
    result = 31 * result + (queryPreTokenizedText ? 1 : 0);
    result = 31 * result + slop;
    return result;
  }

  public static Similarity[] similarities() {
    return new Similarity[]{
        new DefaultSimilarity(),
        new BM25Similarity(),
        new IBSimilarity( new DistributionSPL(), new LambdaTTF(), new NormalizationZ() ), // TODO(gabor) I kind of randomly guessed these
        new LMDirichletSimilarity(),
        new LMJelinekMercerSimilarity(0.5f)
    };
  }

  @SuppressWarnings("ConstantConditions")
  public static LuceneQuerierParams[] all() {
    List<LuceneQuerierParams> lst = new LinkedList<LuceneQuerierParams>();
    for ( Similarity metric : similarities() ) {
      LuceneQuerierParams a = base().withSimilarity(metric);
      for (PhraseSemantics semantics : PhraseSemantics.values()) {
        LuceneQuerierParams b = a.withPhraseSemantics(semantics);
        for (Occur occur : new Occur[]{ Occur.MUST, Occur.SHOULD }) {
          LuceneQuerierParams c = b.withConjunctionMode(occur);
          for (boolean fuzzy : new boolean[]{ false, true }) {
            LuceneQuerierParams d = c.withFuzzy(fuzzy);
            for (boolean entityType : new boolean[]{ false, true }) {
              LuceneQuerierParams e = d.withNERTag(entityType);
              for (boolean slotFill : new boolean[]{ false, true }) {
                LuceneQuerierParams f = e.withSlotFill(slotFill);
                for (boolean slotFillType : new boolean[]{ false, true }) {
                  LuceneQuerierParams g = f.withSlotFillType(slotFillType);
                  for (boolean relation : new boolean[]{ false, true }) {
                    LuceneQuerierParams h = g.withRelation(relation);
                    for (boolean caseSensitive : new boolean[]{ false, true }) {
                      LuceneQuerierParams i = h.withCaseSensitive(caseSensitive);
                      for (boolean usePreTokenizedText: new boolean[] { false, true}) {
                        LuceneQuerierParams j = i.withPreTokenizedText(usePreTokenizedText);
                        if (j.slopIsRelevant()) {
                          for (int slop = 0; slop <= MAX_SLOP; slop++) {
                            LuceneQuerierParams k = j.withSlop(slop);
                            lst.add(k);
                          }
                        } else {
                          lst.add(j);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return lst.toArray(new LuceneQuerierParams[lst.size()]);
  }

  public static Similarity DEFAULT_SIMILARITY = new DefaultSimilarity();

  public static LuceneQuerierParams base() {
    return new LuceneQuerierParams(DEFAULT_SIMILARITY, PhraseSemantics.UNIGRAMS_MUST, Occur.SHOULD, false, false, true, false, false, false,false,0,0,0);
  }

  public static LuceneQuerierParams strict() {
    return new LuceneQuerierParams(DEFAULT_SIMILARITY, PhraseSemantics.PHRASE, Occur.MUST, false, true, true, true, true, true, true,0,0,0);
  }
}
