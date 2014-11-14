package edu.stanford.nlp.kbp.slotfilling.process;

import java.util.*;

import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.logging.Redwood;

/**
 * @author kevinreschke
 */
public abstract class Featurizer {
  
  protected static final Redwood.RedwoodChannels logger = Redwood.channels("Featurize");
  
  /** Build datums from |annotation| for relation mentions headed by |entity|
   *  A relation classifier is supplied for optional filtering. */
  public HashMap<KBPair,SentenceGroup> featurize(Annotation annotation, Maybe<RelationFilter> relationFilter) {
    HashMap<KBPair, SentenceGroup> featurized = new HashMap<>();
    for (Map.Entry<KBPair, Pair<SentenceGroup, List<CoreMap>>> entry : featurizeWithSentences(annotation, relationFilter).entrySet()) {
      featurized.put(entry.getKey(), entry.getValue().first);
    }
    return featurized;
  }
  
  /** Featurize with null relation filter */
  public HashMap<KBPair,SentenceGroup> featurize(Annotation annotation) {
    return featurize(annotation, Maybe.<RelationFilter>Nothing());
  }
  
  
  public Map<KBPair, Pair<SentenceGroup, List<CoreMap>>> featurizeWithSentences(Annotation annotation, Maybe<RelationFilter> relationFilter) {
    HashMap<KBPair,Pair<SentenceGroup,List<CoreMap>>> datums = new HashMap<>();
    for (CoreMap sentence : annotation.get(SentencesAnnotation.class)) {
      for(SentenceGroup sg : featurizeSentence(sentence, relationFilter)) {
        KBPair key = sg.key;
        if (!datums.containsKey(key)) {
          datums.put(key, new Pair<SentenceGroup, List<CoreMap>>(SentenceGroup.empty(key), new ArrayList<CoreMap>()));
        }
        Pair<SentenceGroup, List<CoreMap>> pair = datums.get(key);
        pair.first.merge(sg);
        //noinspection ForLoopReplaceableByForEach
        for (int i = 0; i < sg.size(); ++i) { pair.second.add(sentence); }
        assert pair.first.size() == pair.second.size();
      }
    }
    return datums;
  }
  
  /**
   * Build datums for relations found in |sentence| and headed by |entity|.
   * 
   * @param sentence The sentence to featurize
   * @param relationFilter  Optional relation filter for within-sentence filtering
   * @return List of singleton sentence groups (each with a single datum).  
   */
  public abstract List<SentenceGroup> featurizeSentence(CoreMap sentence, Maybe<RelationFilter> relationFilter);
}
