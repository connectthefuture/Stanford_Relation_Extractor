package edu.stanford.nlp.kbp.slotfilling.ir;

import edu.stanford.nlp.kbp.common.*;

import java.util.*;
import java.io.Serializable;

import static edu.stanford.nlp.util.logging.Redwood.Util.warn;

/**
 * A representation of the knowledge base.
 */
public class KnowledgeBase implements Serializable {

  public final LinkedHashMap<KBPEntity, LinkedHashSet<KBPSlotFill>> data;
  public final Map<String, Set<KBPSlotFill>> dataByName;

  public KnowledgeBase() {
    data = new LinkedHashMap<>();
    dataByName = new HashMap<>();
  }

  /** Check whether the knowledge base has this (entity, relation, slotValue) triple */
  public boolean contains(KBTriple triple) {
    for (KBPSlotFill fill : data.get(triple.getEntity())) {
      if (fill.key.equals(triple)) { return true; }
    }
    return false;
  }

  /** Check whether the knowledge base has this (entity, slotValue) pair, for any relation */
  public boolean contains(KBPair pair) {
    LinkedHashSet<KBPSlotFill> fills = data.get(pair.getEntity());
    if (fills == null) {
      return false;
    }
    for (KBPSlotFill fill : fills) {
      if (pair.equals(fill.key)) { return true; }
    }
    return false;
  }

  /** Check whether the knowledge base has this slot fill; this reduces to whether the knowledge base has the slot fill's triple */
  public boolean contains(KBPSlotFill queryFill) {
    return contains(queryFill.key);
  }

  public void put(List<KBPSlotFill> facts) {
    for( KBPSlotFill fact : facts ) {
      put(fact);
    }
  }
  public void put(KBPSlotFill fact) {
    KBPEntity entity = fact.key.getEntity();
    if(!data.containsKey(entity)) {
      data.put(entity, new LinkedHashSet<KBPSlotFill>());
      dataByName.put(entity.name, new HashSet<KBPSlotFill>());
    }
    data.get(entity).add(fact);
    dataByName.get(entity.name).add(fact);
  }

  public Maybe<Set<KBPSlotFill>> get(KBPEntity entity) {
    if( data.containsKey(entity) ) {
      return Maybe.Just((Set<KBPSlotFill>) data.get(entity));
    } else if (dataByName.containsKey(entity.name)) {
      return Maybe.Just(dataByName.get(entity.name));
    } else {
      return Maybe.Nothing();
    }
  }

  public boolean isEmpty() {
    return data.isEmpty();
  }

  public List<KBTriple> triples() {
    return triples(Integer.MAX_VALUE);
  }
  public List<KBTriple> triples(int maxTriples) {
    List<KBTriple> triples = new ArrayList<>();
    int numTr = 0;
    for (LinkedHashSet<KBPSlotFill> fills : data.values()) {
      for (KBPSlotFill fill : fills) {
        assert this.contains(fill.key);
        numTr++;
        if(numTr > maxTriples)
          break;
        triples.add(fill.key);
      }
    }
    Collections.sort(triples);
    return triples;
  }
}
