package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.logging.Redwood;
import edu.washington.cs.knowitall.extractor.ReVerbExtractor;
import edu.washington.cs.knowitall.extractor.conf.ConfidenceFunction;
import edu.washington.cs.knowitall.extractor.conf.ReVerbOpenNlpConfFunction;
import edu.washington.cs.knowitall.nlp.ChunkedSentence;
import edu.washington.cs.knowitall.nlp.OpenNlpSentenceChunker;
import edu.washington.cs.knowitall.nlp.extraction.ChunkedBinaryExtraction;
import edu.washington.cs.knowitall.normalization.BinaryExtractionNormalizer;
import edu.washington.cs.knowitall.normalization.NormalizedBinaryExtraction;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * Add relations to an entity graph from a document using Reverb
 */
public class ReverbRelationExtractor extends OpenIERelationExtractor {
  private static Redwood.RedwoodChannels logger = Redwood.channels("ReVerb");

  protected static OpenNlpSentenceChunker chunker;
  protected static ReVerbExtractor reverb;
  protected static ConfidenceFunction confFunc;
  protected static BinaryExtractionNormalizer binNormalizer;

  public ReverbRelationExtractor() {
    try {
      chunker = new OpenNlpSentenceChunker();
      reverb = new ReVerbExtractor();
      //reverb.setAllowUnary(true);
      confFunc = new ReVerbOpenNlpConfFunction();
      binNormalizer = new BinaryExtractionNormalizer();
    } catch (IOException e) {
      err( "Could not load ReVerb");
      fatal(e);
    }
  }

  /**
   * Extract relations using ReVerb
   * @param doc A document, already passed through {@link edu.stanford.nlp.kbp.slotfilling.ir.PostIRAnnotator} and
   *            the various Mention Annotators (e.g., {@link edu.stanford.nlp.kbp.slotfilling.process.SlotMentionAnnotator}).
   * @return A list of slot fills corresponding to the slots extracted by this extractor.
   *         Note, importantly, that the relations in these slot fills do <b>not</b> have to be proper
   *         KBP relations.
   */
  @Override
  public List<KBPSlotFill> extractRelations(Annotation doc) {
    String docId = doc.get(CoreAnnotations.DocIDAnnotation.class);
    String indexName = doc.get(KBPAnnotations.SourceIndexAnnotation.class);

    List<KBPSlotFill> fills = new ArrayList<>();

    for( CoreMap sentence : doc.get(CoreAnnotations.SentencesAnnotation.class) ) {
      if (sentence.get(CoreAnnotations.TokensAnnotation.class).size() > 100) { continue; }
      List<EntityMention> mentions = new ArrayList<>();
      if (sentence.get(KBPAnnotations.SlotMentionsAnnotation.class) != null) {
        mentions.addAll( sentence.get(KBPAnnotations.SlotMentionsAnnotation.class) );
      }
      if (sentence.get(MachineReadingAnnotations.EntityMentionsAnnotation.class) != null) {
        mentions.addAll(sentence.get(MachineReadingAnnotations.EntityMentionsAnnotation.class));
      }
      if(mentions.size() == 0) { continue; }  // no slot mentions in this sentence, so no point extracting anything

      // Prepare the sentence for ReVerb
      ChunkedSentence reverbSentence = chunker.chunkSentence(
          sentence.containsKey(CoreAnnotations.OriginalTextAnnotation.class) ?  // prefer OriginalText (should always exist)
              sentence.get(CoreAnnotations.OriginalTextAnnotation.class) :
                sentence.containsKey(CoreAnnotations.TextAnnotation.class) ?    // ... then Text
                sentence.get(CoreAnnotations.TextAnnotation.class) :
                CoreMapUtils.sentenceToMinimalString(sentence));                // ... then give up and convert it to text manually

      // Run ReVerb
      for(ChunkedBinaryExtraction extr : reverb.extract(reverbSentence)) {
        // Normalize ReVerb extraction
        NormalizedBinaryExtraction nExtr = binNormalizer.normalize(extr);

        // Find the actual token offsets of the left argument
        for ( Span span1 : Utils.getTokenSpan(sentence.get(CoreAnnotations.TokensAnnotation.class),
            StringUtils.join(extr.getArgument1().getTokens(), ""),
            Maybe.Just(new Span(extr.getArgument1().getRange().getStart(), extr.getArgument1().getRange().getEnd())))) {
          // Find the actual token offsets of the right argument
          for ( Span span2 : Utils.getTokenSpan(sentence.get(CoreAnnotations.TokensAnnotation.class),
              StringUtils.join(extr.getArgument2().getTokens(), ""),
              Maybe.Just(new Span(extr.getArgument2().getRange().getStart(), extr.getArgument2().getRange().getEnd())))) {
            // Ensure that at least one of the extraction's arguments is a valid "slot mention"
            for( Pair<KBPEntity, KBPEntity> pair : alignExtractions(sentence, mentions, span1, span2) ) {
              double conf = confFunc.getConf(nExtr);
              KBPRelationProvenance provenance =
                new KBPRelationProvenance(docId, indexName, sentence.get(CoreAnnotations.SentenceIndexAnnotation.class), span1, span2, sentence);
              fills.add(
                  KBPNew.from(pair.first).slotValue(pair.second).rel(nExtr.getRelationNorm().toString())
                      .provenance(provenance)
                      .score(conf).KBPSlotFill()
              );
            }
          }
        }
      }
    }

    // Print results
    startTrack("Reverb additions for " + doc.get(CoreAnnotations.DocIDAnnotation.class));
    for (KBPSlotFill fill : fills) {
      logger.debug(fill.key);
    }
//    if (fills.size() > 0) { logger.log(fills.size() + " extractions from Reverb added."); }
    endTrack("Reverb additions for " + doc.get(CoreAnnotations.DocIDAnnotation.class));

    return fills;
  }

  /**
   * Heuristically align our ReVerb spans to some known mention.
   *
   * @param sentence The sentence we have extracted relations from.
   * @param mentions The mentions in the sentence, as extracted by {@link edu.stanford.nlp.kbp.slotfilling.process.SlotMentionAnnotator}.
   * @param leftArgSpan The span of the left argument from ReVerb, aligned to CoreNLP tokens.
   * @param rightArgSpan The span of the right argument from ReVerb, aligned to CoreNLP tokens.
   *
   * @return A pair of entities corresponding to the two arguments from the ReVerb extraction.
   */
  private Maybe<Pair<KBPEntity, KBPEntity>> alignExtractions(
      CoreMap sentence,
      List<EntityMention> mentions,
      Span leftArgSpan,
      Span rightArgSpan ) {

    // Variables we need to fill
    KBPEntity leftEntity = null;
    KBPEntity rightEntity = null;
    double leftOverlap  = 0.74;
    double rightOverlap = 0.74;

    // Get the entities and types
    for( EntityMention mention : mentions ) {
      double leftOverlapForMention = smartOverlap(sentence, leftArgSpan, mention.getExtent());
      double rightOverlapForMention = smartOverlap(sentence, rightArgSpan, mention.getExtent());
      if (leftOverlapForMention > leftOverlap && rightOverlapForMention > rightOverlap) {
        // Case: both arguments overlap with the mention
        // Action: take the higher overlap
        if (leftOverlapForMention > rightOverlapForMention) {
          leftEntity = rewriteCanonicalMention(sentence, mention);
          leftOverlap = leftOverlapForMention;
        } else {
          rightEntity = rewriteCanonicalMention(sentence, mention);
          rightOverlap = rightOverlapForMention;
        }
      } else if (leftOverlapForMention > leftOverlap) {
        // Case: left argument overlap
        leftEntity = rewriteCanonicalMention(sentence, mention);
        leftOverlap = leftOverlapForMention;
      } else if (rightOverlapForMention > rightOverlap) {
        // Case: right argument overlaps
        rightEntity = rewriteCanonicalMention(sentence, mention);
        rightOverlap = rightOverlapForMention;
      }
    }

    // Return any matches
    if (leftEntity != null && rightEntity != null && !leftEntity.equals(rightEntity)) {
      //noinspection SuspiciousNameCombination
      return Maybe.Just(Pair.makePair(leftEntity, rightEntity));
    } else {
      return Maybe.Nothing();
    }
  }

  /**
   * A weighting of POS tags in terms of their importance for two {@link Span}s overlapping.
   * @see ReverbRelationExtractor#smartOverlap(edu.stanford.nlp.util.CoreMap, edu.stanford.nlp.ie.machinereading.structure.Span, edu.stanford.nlp.ie.machinereading.structure.Span)
   */
  private static Counter<String> posWeighting = new ClassicCounter<String>(){{
    setCount("CC",   0.10);
    setCount("CD",   0.15);
    setCount("DT",   0.01);
    setCount("EX",   0.01);
    setCount("FW",   0.10);
    setCount("IN",   0.10);
    setCount("JJ",   0.50);
    setCount("JJR",  0.50);
    setCount("JJS",  0.50);
    setCount("LS",   0.10);
    setCount("MD",   0.10);
    setCount("NN",   0.50);
    setCount("NNS",  0.50);
    setCount("NNP",  0.50);
    setCount("NNPS", 0.50);
    setCount("PDT",  0.10);
    setCount("POS",  0.10);
    setCount("PRP",  0.01);
    setCount("PRP$", 0.01);
    setCount("RB",   0.20);
    setCount("RBR",  0.20);
    setCount("RBS",  0.20);
    setCount("RP",   0.20);
    setCount("SYM",  0.01);
    setCount("TO",   0.10);
    setCount("UH",   0.01);
    setCount("VB",   0.3);
    setCount("VBD",  0.3);
    setCount("VBG",  0.3);
    setCount("VBN",  0.3);
    setCount("VBP",  0.3);
    setCount("VBZ",  0.3);
    setCount("WDT",  0.1);
    setCount("WP",   0.1);
    setCount("WP$",  0.1);
    setCount("WRB",  0.1);
  }};
  /**
   * A weighting of NER tags in terms of their importance for two {@link Span}s overlapping.
   * @see ReverbRelationExtractor#smartOverlap(edu.stanford.nlp.util.CoreMap, edu.stanford.nlp.ie.machinereading.structure.Span, edu.stanford.nlp.ie.machinereading.structure.Span)
   */
  private static Counter<NERTag> nerWeighting = new ClassicCounter<NERTag>(){{
    setCount(NERTag.CAUSE_OF_DEATH,    0.50);
    setCount(NERTag.CITY,              0.75);
    setCount(NERTag.COUNTRY,           0.75);
    setCount(NERTag.CRIMINAL_CHARGE,   0.50);
    setCount(NERTag.DATE,              0.75);
    setCount(NERTag.IDEOLOGY,          0.50);
    setCount(NERTag.LOCATION,          0.75);
    setCount(NERTag.MISC,              0.40);
    setCount(NERTag.MODIFIER,          0.40);
    setCount(NERTag.NATIONALITY,       0.70);
    setCount(NERTag.NUMBER     ,       0.01);
    setCount(NERTag.ORGANIZATION,      1.00);
    setCount(NERTag.PERSON,            1.00);
    setCount(NERTag.RELIGION,          0.50);
    setCount(NERTag.STATE_OR_PROVINCE, 0.75);
    setCount(NERTag.TITLE,             0.80);
    setCount(NERTag.URL,               0.75);
    setCount(NERTag.DURATION,          0.01);
  }};

  /**
   * Get a weighted overlap between two spans, taking into consideration that some types of words
   * are more important to match than others.
   * This serves two purposes: it ensures that "better" matches get prioritized even if they're shorter,
   * and it ensures that the minimum text overlap threshold works better than if it were a simple threshold
   * on length (e.g., we don't care if extra determiners are thrown in, but care a lot if extra nouns are).
   *
   * @param sentence The sentence the spans are over.
   * @param lowPrecisionSpan The low precision span, canonically from ReVerb.
   * @param highPrecisionSpan The high precision span, canonically from {@link edu.stanford.nlp.kbp.slotfilling.process.SlotMentionAnnotator}.
   *
   * @return A weighted overlap, between 0 and 1, corresponding to how well the two spans match.
   */
  private double smartOverlap(CoreMap sentence, Span lowPrecisionSpan, Span highPrecisionSpan) {
    if (highPrecisionSpan.equals(lowPrecisionSpan)) { return 1.0; }
    double totalPossible = 0.0;
    double sumOverlap = 0.0;
    List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
    // Iterate over sentence to compute overlap
    for (int i = 0; i < tokens.size(); ++i) {
      String pos = tokens.get(i).tag();
      // Don't care about clauses (e.g., "a colonel <of the 3rd New York Regiment>") in ReVerb.
      if (lowPrecisionSpan.contains(i) && !highPrecisionSpan.contains(i) && pos.equals("IN")) { break; }
      // Accumulate overlap
      if (highPrecisionSpan.contains(i) || lowPrecisionSpan.contains(i)) {
        // Get weighting for overlap
        double possible;
        String ner = tokens.get(i).ner();
        NERTag tag;
        if (ner != null && (tag = NERTag.fromShortName(ner).orNull()) != null && tag != NERTag.NUMBER && tag != NERTag.DURATION) {
          possible = nerWeighting.getCount(tag);
        } else {
          possible = posWeighting.getCount(pos);
        }
        // Run weighting
        totalPossible += possible;
        if (highPrecisionSpan.contains(i) && lowPrecisionSpan.contains(i)) {
          sumOverlap += possible;
        }
      }
    }
    // Return
    if (totalPossible == 0.0) { return 0.0; }
    return sumOverlap / totalPossible;
  }

  /**
   * A helper function to get the canonical form of an entity, resolving coreference.
   * @param sentence The containing sentence, for context.
   * @param mention The entity in question, as a mention.
   * @return A KBPEntity corresponding to the canonicalization of the entity mention.
   */
  private KBPEntity rewriteCanonicalMention(CoreMap sentence, EntityMention mention) {
    Span entitySpan = mention.getExtent();
    List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
    Maybe<String> antecedent = Maybe.Nothing();

    for (int i = entitySpan.start(); i < entitySpan.end(); i++) {
      antecedent = antecedent.orElse(Maybe.fromNull(tokens.get(i).get(CoreAnnotations.AntecedentAnnotation.class)));
    }
    KBPEntity entity = KBPNew.from(Utils.getKbpEntity(mention)).entName(antecedent.getOrElse(mention.getFullValue())).KBPEntity();
    assert entity.type == Utils.getNERTag(mention).orNull();
    return entity;
  }

}
