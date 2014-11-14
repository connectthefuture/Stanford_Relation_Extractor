package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.kbp.common.KBPNew;
import edu.stanford.nlp.kbp.common.NERTag;
import edu.stanford.nlp.kbp.common.RelationType;
import org.junit.Before;
import org.junit.Test;

import java.util.LinkedList;
import java.util.List;

import static junit.framework.Assert.*;

/**
 * A quick test for the portions of the Probabilities class
 * which do not require the broader KBP infrastructure (e.g.,
 * making IR queries)
 *
 * @author Gabor Angeli
 */
public class ProbabilitiesTest {

  private Probabilities probabilities;

  @Before
  public void setUp() {
    List<String> allSlotFills = new LinkedList<String>();
    allSlotFills.add("Canada");
    this.probabilities = new Probabilities(null, allSlotFills, Math.random()); // Test should be invariant to these args
  }

  @Test
  public void testNERTagPriors() {
    assertEquals(1.0, Probabilities.ofRelationTypeORGANIZATION + Probabilities.ofRelationTypePERSON, 1e-5);
  }

  @Test
  public void testProbabilityOfRelation() {
    // Let us know if these are changed by someone
    assertEquals(0.0223444134627622040,
        probabilities.ofRelation(RelationType.PER_COUNTRY_OF_BIRTH));
    // Make sure the probabilities sum to 1.0
    double sum = 0.0;
    for (RelationType rel : RelationType.values()) {
      sum += probabilities.ofRelation(rel);
    }
    assertEquals(1.0, sum, 1e-5);
  }

  @Test
  public void testProbabilityOfRelationGivenEntity() {
    // Let us know if something changed
    assertEquals(0.0223444134627622040 / Probabilities.ofRelationTypePERSON,
        probabilities.ofRelationGivenEntity(RelationType.PER_COUNTRY_OF_BIRTH, KBPNew.entName("Lennon").entType(NERTag.PERSON).KBPOfficialEntity()));
    // Make sure the probabilities sum to 1.0 (person)
    double sum = 0.0;
    for (RelationType rel : RelationType.values()) {
      if (rel.entityType == NERTag.PERSON) {
        sum += probabilities.ofRelationGivenEntity(rel, KBPNew.entName("Lennon").entType(rel.entityType).KBPOfficialEntity());
      }
    }
    assertEquals(1.0, sum, 1e-5);
    // Make sure the probabilities sum to 1.0 (organization)
    sum = 0.0;
    for (RelationType rel : RelationType.values()) {
      if (rel.entityType == NERTag.ORGANIZATION) {
        sum += probabilities.ofRelationGivenEntity(rel, KBPNew.entName("Apple").entType(rel.entityType).KBPOfficialEntity());
      }
    }
    assertEquals(1.0, sum, 1e-5);

  }
}
