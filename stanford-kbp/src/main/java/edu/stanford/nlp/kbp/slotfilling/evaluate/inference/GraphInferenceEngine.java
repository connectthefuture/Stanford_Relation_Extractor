package edu.stanford.nlp.kbp.slotfilling.evaluate.inference;


import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.evaluate.EntityGraph;
import edu.stanford.nlp.util.Pair;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Inference rules
 */
public abstract class GraphInferenceEngine {

  /**
   * Run inference on the given entity graph, given a pivot entity.
   *
   * @param graph The graph to run inference over.
   * @param entity The pivot entity, for whom we are primarily interested in finding new slots.
   * @return The same entity graph, but with additional slot values inferred.
   */
  public abstract EntityGraph apply(EntityGraph graph, KBPEntity entity);

  /**
   * Prune some of the relations added to the graph, if they have no chance of being useful in inference.
   *
   * @param reln The string form of the relation to potentially prune while constructing the entity graph.
   * @return True if this relation should not be added to the entity graph, as it will never be useful in any inference.
   */
  public boolean isUsefulRelation(String reln) {
    return true;
  }

  /**
   * Strip type information from the Tuffy relations.
   * The type information is there originally to distinguish between the same relation with
   * different types; however, this means that the Tuffy relations would no longer align with
   * the KBP relation strings -- this function can be used to convert from Tuffy relation strings
   * (with types) to the KBP equivalent (without types).
   *
   * @param typedRelation The Tuffy typed relation
   * @return The KBP relation, without types, corresponding to the Tuffy relation.
   */
  protected static String untypedRelation(String typedRelation) {
    int indexOfTyping = typedRelation.indexOf("_TYPE_");
    String rtn = typedRelation;
    if (indexOfTyping >= 0) {
      rtn = typedRelation.substring(0, indexOfTyping);
    }
    if (rtn.startsWith("per_")) {
      rtn = "per:" + rtn.substring(4);
    }
    if (rtn.startsWith("org_")) {
      rtn = "org:" + rtn.substring(4);
    }
    //noinspection LoopStatementThatDoesntLoop
    for (RelationType rel : RelationType.fromString(rtn)) {
      return rel.canonicalName;
    }
    return rtn.replaceAll("_", " ");
  }

  protected static Pair<NERTag, NERTag> getTypes(String typedRelation) {
    Matcher rgx = Pattern.compile(".*_TYPE_([A-Z]+)_TO_([A-Z]+).*").matcher(typedRelation);
    if( rgx.matches() ) {
      return Pair.makePair(NERTag.fromString(rgx.group(1)).get(), NERTag.fromString(rgx.group(2)).get());
    }
    throw new IllegalArgumentException("Invalid type string: " + typedRelation);
  }

  public static String cleanRelation(String rawRelation) {
    return rawRelation.replaceAll("\\s+", "_").replaceAll("[^a-zA-Z0-9_]", "_" );
  }
  public static String cleanRelation(String rawRelation, NERTag type1, NERTag type2) {
    return cleanRelation(rawRelation) + "_TYPE_" + type1.shortName + "_TO_" + type2.shortName;
  }
  public static String cleanRelation(KBPSlotFill fill) {
    return cleanRelation(fill.key.relationName, fill.key.entityType, fill.key.slotType.getOrElse(NERTag.MISC));
  }
  public static String cleanEntity(String entity, NERTag type) {
    return type.shortName.toUpperCase() + "_" + entity.replaceAll("\\s+","_").replaceAll("[^a-zA-Z0-9_]", "_");
  }
  public static String cleanEntity(KBPEntity entity) {
    return cleanEntity(entity.name, entity.type);
  }

}
