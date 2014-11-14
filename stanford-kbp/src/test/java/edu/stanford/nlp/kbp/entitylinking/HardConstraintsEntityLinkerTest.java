package edu.stanford.nlp.kbp.entitylinking;

import edu.stanford.nlp.kbp.common.EntityContext;
import edu.stanford.nlp.kbp.common.KBPNew;
import edu.stanford.nlp.kbp.common.NERTag;
import org.junit.Test;

import static junit.framework.Assert.*;

/**
 * Test the hard constraints enforced by the entity linker.
 *
 * @author Gabor Angeli
 */
public class HardConstraintsEntityLinkerTest {

  private EntityLinker linker = new EntityLinker.HardConstraintsEntityLinker();

  @Test
  public void testTypeCheck() {
    EntityContext a = new EntityContext(KBPNew.entName("Samsung").entType(NERTag.PERSON).KBPEntity());
    EntityContext aAgain = new EntityContext(KBPNew.entName("Samsung").entType(NERTag.PERSON).KBPEntity());
    EntityContext b = new EntityContext(KBPNew.entName("Samsung").entType(NERTag.ORGANIZATION).KBPEntity());
    assertTrue(linker.sameEntity(a, aAgain));
    assertFalse(linker.sameEntity(a, b));
  }

  @Test
  public void testAcronymCheck() {
    EntityContext a = new EntityContext(KBPNew.entName("Ben & Jerry").entType(NERTag.ORGANIZATION).KBPEntity());
    EntityContext aAcr = new EntityContext(KBPNew.entName("B&J").entType(NERTag.ORGANIZATION).KBPEntity());
    EntityContext aAcr2 = new EntityContext(KBPNew.entName("BJ").entType(NERTag.ORGANIZATION).KBPEntity());
    EntityContext aAcrNeg = new EntityContext(KBPNew.entName("BP").entType(NERTag.ORGANIZATION).KBPEntity());
    EntityContext aAcrNeg2 = new EntityContext(KBPNew.entName("B P").entType(NERTag.ORGANIZATION).KBPEntity());
    assertTrue(linker.sameEntity(a, aAcr));
    assertTrue(linker.sameEntity(a, aAcr2));
    assertFalse(linker.sameEntity(a, aAcrNeg));
    assertFalse(linker.sameEntity(a, aAcrNeg2));
  }
}
