package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;

import java.util.Arrays;
import java.util.function.Function;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static edu.stanford.nlp.util.logging.Redwood.Util.log;

/**
 * A base class for some heuristic, rule-based relation extractors.
 *
 * @author Gabor Angeli
 */
public abstract class HeuristicRelationExtractor extends RelationClassifier {
  protected static final Redwood.RedwoodChannels logger = Redwood.channels("HeuristicRelationExtractor");

  public abstract Collection<Pair<String, Integer>> extractRelations(KBPair key, CoreMap[] input);

  @Override
  public Counter<Pair<String,Maybe<KBPRelationProvenance>>> classifyRelations(SentenceGroup input, Maybe<CoreMap[]> rawSentences) {
    Counter<Pair<String, Maybe<KBPRelationProvenance>>> rels = new ClassicCounter<>();
    for (CoreMap[] sentences : rawSentences) {
      for (Pair<String, Integer> rel : extractRelations(input.key, sentences)) {
        Span entitySpan = new Span();
        Span slotSpan = new Span();
        int i = -1;
        for(CoreLabel l : sentences[rel.second].get(CoreAnnotations.TokensAnnotation.class)){
          i++;
          Boolean entity = l.get(KBPAnnotations.IsEntity.class);

          if(entity != null && entity){
            if(i < entitySpan.start()){
              entitySpan.setStart(i);
            }
            if( (i+1) > entitySpan.end()){
              entitySpan.setEnd(i+1);
            }
          }
          Boolean slot = l.get(KBPAnnotations.IsSlot.class);
          if(slot != null && slot){
            if(i < slotSpan.start()){
              slotSpan.setStart(i);
            }
            if((i+1) > slotSpan.end()){
              slotSpan.setEnd(i+1);
            }
          }
        }
        boolean valid = true;
        if(entitySpan.start() > entitySpan.end() || slotSpan.start() > slotSpan.end()) {
          valid = false;
        }

        if (valid) {
          Maybe<KBPRelationProvenance> provenance = KBPRelationProvenance.computeFromSpans(sentences[rel.second], entitySpan, slotSpan);
          provenance.get().setClassifierClass(HeuristicRelationExtractor.class);

          logger.log("MATCH for relation " + rel.first() + " with and " +( provenance.get().isOfficial() ? " official ": "UNofficial" )+ " sentence " + StringUtils.joinWithOriginalWhiteSpace(sentences[rel.second()].get(CoreAnnotations.TokensAnnotation.class)));
          rels.setCount(Pair.makePair(rel.first, provenance), Double.POSITIVE_INFINITY);
        } else {
//          throw new RuntimeException("how come not able to make provenance from sentence");
          rels.setCount(Pair.makePair(rel.first, KBPRelationProvenance.compute(sentences[rel.second], KBPNew.from(input.key).rel(rel.first).KBTriple())), Double.POSITIVE_INFINITY);
        }
      }
    }

    return rels;
  }

  @Override
  public TrainingStatistics train(KBPDataset<String, String> trainSet) {
    // NOOP
    log("training for a heuristic relation classifier is a noop");
    return TrainingStatistics.empty();
  }

  @Override
  public void load(ObjectInputStream in) throws IOException, ClassNotFoundException {
    // NOOP
    assert in == null;
  }

  @Override
  public void save(ObjectOutputStream out) throws IOException {
    // NOOP
  }


  @Deprecated
  public static final Function<Pair<KBPair, CoreMap[]>, Counter<Pair<String, Maybe<KBPRelationProvenance>>>> allExtractors;
  static {
    
    // Extractors to run
    List<HeuristicRelationExtractor> list = new ArrayList<HeuristicRelationExtractor>();
    if(Props.TRAIN_TOKENSREGEX_DIR != null)
      list.add(new TokensRegexExtractor());
    
    if(Arrays.asList(Props.TEST_AUXMODELS).contains(ModelType.SEMGREX))
      list.add(new SemgrexExtractor());
    
    final HeuristicRelationExtractor[] extractors = list.toArray(new HeuristicRelationExtractor[0]);
    
    // Set the allExtractors variable
    allExtractors = in -> {
      Counter<Pair<String, Maybe<KBPRelationProvenance>>> extractions = new ClassicCounter<Pair<String, Maybe<KBPRelationProvenance>>>();

      for (HeuristicRelationExtractor extractor : extractors) {
        logger.log("Extractors are " + extractor.getClass());
        for (Pair<String, Integer> rel : extractor.extractRelations(in.first, in.second)) {
          extractions.setCount(Pair.makePair(rel.first,
              KBPRelationProvenance.compute(in.second[rel.second], KBPNew.from(in.first).rel(rel.first).KBTriple())), 1.0);
        }
      }
      return extractions;
    };
  }
}
