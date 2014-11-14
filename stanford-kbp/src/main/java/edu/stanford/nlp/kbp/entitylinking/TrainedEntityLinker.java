package edu.stanford.nlp.kbp.entitylinking;

import edu.stanford.nlp.classify.Classifier;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.kbp.common.EntityContext;
import edu.stanford.nlp.kbp.common.Maybe;
import edu.stanford.nlp.kbp.common.NERTag;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

import static edu.stanford.nlp.util.logging.Redwood.log;

/**
 *
 * @author Melvin Jose
 */
public class TrainedEntityLinker extends EntityLinker implements Serializable{

  private static final long serialVersionUID = 1L;
  //redwood
  private static final Redwood.RedwoodChannels logger = Redwood.channels("TrainedEntLinker");
  //Create classifier variables
  Map<NERTag, Classifier<Boolean, Feature>> classifiers;
  EntityLinkingFeaturizer featurizer;

  public TrainedEntityLinker(Map<NERTag, Classifier<Boolean, Feature>> classifiers, EntityLinkingFeaturizer featurizer) {
    this.classifiers = classifiers;
    this.featurizer = featurizer;
  }

  @Override
  public Maybe<String> link(EntityContext context) {
    return Maybe.Nothing();
  }

  @Override
  protected boolean sameEntityWithoutLinking(EntityContext entityOne, EntityContext entityTwo) {
   //((LinearClassifier) classifier).justificationOf(featurizer.featurize(Pair.makePair(entityOne, entityTwo), false));
    RVFDatum<Boolean, Feature> features = featurizer.featurize(Pair.makePair(entityOne, entityTwo), false);
    if(entityOne.entity.type.equals(entityTwo.entity.type)) {
      if(classifiers.get(entityOne.entity.type) != null) {
        if(classifiers.get(entityOne.entity.type).classOf(features)) {
          logger.log(entityOne.entity.name+"\t"+entityTwo.entity.name);
          logger.prettyLog(features);
          return true;
        }
        return false;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }

  public void printJustification(EntityContext entityOne, EntityContext entityTwo) {
    ((LinearClassifier) classifiers.get(entityOne.entity.type)).justificationOf(featurizer.featurize(Pair.makePair(entityOne, entityTwo), false));
  }

}
