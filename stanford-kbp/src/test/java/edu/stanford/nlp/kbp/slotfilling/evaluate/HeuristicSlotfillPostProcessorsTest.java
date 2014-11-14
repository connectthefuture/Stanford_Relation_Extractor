package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.entitylinking.EntityLinker;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.MetaClass;
import edu.stanford.nlp.util.Pair;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import java.util.*;
import java.util.regex.Matcher;

import static junit.framework.Assert.*;

/**
 * A test for the Heuristic slot fill post processors,
 * testing basic functionality.
 *
 * @author Gabor Angeli
 */
public class HeuristicSlotfillPostProcessorsTest extends PostProcessorsData {

  @SuppressWarnings("unchecked")
  private Map<KBPEntity, List<KBPSlotFill>> regression1Data() {
    Map<KBPEntity, List<KBPSlotFill>> data = new HashMap<KBPEntity, List<KBPSlotFill>>();

    String[] sentenceGloss = "The Ferrari pilot beat home Briton Lewis Hamilton while reigning champion".split("\\s+");
    String[] posGloss = "DT NNP NN VBD NN NNP NNP NNP IN VBG NN".split("\\s+");
    String[] nerGloss = "O O TITLE O O NATIONALITY PERSON PERSON O O TITLE".split("\\s+");
    CoreMap sentence = mkCoreMap(sentenceGloss, posGloss, nerGloss);

    data.put(hamilton, new ArrayList<KBPSlotFill>());
    data.get(hamilton).add(fill(hamilton, RelationType.PER_TITLE, "pilot", NERTag.TITLE, 0.8, 6, 7, 2, 3, sentence));

    return data;
  }

  @Before
  public void setUp() throws Exception {
    Props.TEST_CONSISTENCY_REWRITE = true;
    Props.ENTITYLINKING_LINKER = Lazy.from((EntityLinker) MetaClass.create(Props.KBP_ENTITYLINKER_CLASS).createInstance());
  }

  @Test
  public void testGetData() {
    assertEquals(2, dummyData().keySet().size());
    assertEquals(9, dummyData().get(julie).size());
    assertEquals(2, dummyData().get(stanford).size());
  }

  @Test
  public void testGetCornerCaseData() {
    assertEquals(1, orderingCornerCasesData().keySet().size());
    assertEquals(5, orderingCornerCasesData().get(julie).size());
  }

  @Test
  public void testOrderingPairwise() {
    Map<KBPEntity, List<KBPSlotFill>> filtered = new HeuristicSlotfillPostProcessors.DuplicateRelationOnlyInListRelations().postProcess(orderingCornerCasesData());
    assertEquals(2, filtered.get(julie).size());
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", 0.8)));  // right
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_AGE, "Canada", 0.7)));               // wrong, but no way to know why
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "21", 0.6)));
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.PER_AGE, "21", 0.65)));
  }

  @Test
  public void testOrderingHoldOneOut() {
    Map<KBPEntity, List<KBPSlotFill>> filtered = new HeuristicSlotfillPostProcessors.RespectDeclaredIncompatibilities().postProcess(orderingCornerCasesData());
    assertEquals(3, filtered.get(julie).size());
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", 0.8))); // right
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "France", 0.7))); // wrong, but no way to know why
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_AGE, "21", 0.65)));                 // right
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "21", 0.6)));
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.PER_AGE, "Canada", 0.6)));
  }

  @Test
  public void testOrderingCornerCase() {
    Map<KBPEntity, List<KBPSlotFill>> filtered =
        new HeuristicSlotfillPostProcessors.DuplicateRelationOnlyInListRelations().and(
        new HeuristicSlotfillPostProcessors.RespectDeclaredIncompatibilities()).postProcess(orderingCornerCasesData());
    assertEquals(2, filtered.get(julie).size());
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", 0.8)));  // right
    // vv
    //    We miss this if we order things naively.
    //    Canada is the top choice for both AGE and COUNTRY_OF_BIRTH, and the duplicate relation
    //    filter will make sure it's guessed for both. Then, the slot consistency filter will drop
    //    PER_AGE: Canada since it's incompatible with PER_COUNTRY_OF_BIRTH -- but, it has in the
    //    process also forgotten that PER_AGE: 21 would still have been consistent.
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_AGE, "21", 0.65)));                  // right
    // ^^
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "France", 0.7)));
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "21", 0.6)));
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.PER_AGE, "Canada", 0.6)));
  }

  @Test
  public void testIgnoreScoreInEquals() {
    assertTrue(dummyData().get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", 1.0)));
  }

  @Test
  public void testRespectRelationType() {
    Map<KBPEntity, List<KBPSlotFill>> filtered = new HeuristicSlotfillPostProcessors.RespectRelationTypes().postProcess(dummyData());
    assertEquals(7, filtered.get(julie).size());
    assertEquals(1, filtered.get(stanford).size());
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", 0.8)));
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.PER_ALTERNATE_NAMES, "Canada", 0.8)));
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.ORG_MEMBERS, "Chris Manning", 0.8)));
  }

  @Test
  public void testNoDuplicates() {
    Map<KBPEntity, List<KBPSlotFill>> filtered = new HeuristicSlotfillPostProcessors.NoDuplicates().postProcess(dummyData());
    assertEquals(7, filtered.get(julie).size());
    assertEquals(2, filtered.get(stanford).size());
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", 0.8)));
  }

  @Ignore
  @Test
  public void testNoDuplicatesApproximate() {
    Map<KBPEntity, List<KBPSlotFill>> filtered = new HeuristicSlotfillPostProcessors.NoDuplicatesApproximate().postProcess(approximateDuplicateData());

    assertTrue( filtered.get(julie).contains(fill(julie, RelationType.PER_SIBLINGS, "Adan Chavez", 1.0)));
    assertFalse( filtered.get(julie).contains(fill(julie, RelationType.PER_SIBLINGS, "Adan", 0.9)) );
    assertTrue( filtered.get(julie).contains(fill(julie, RelationType.PER_TITLE, "singer/songwriter", 1.0)));
    assertFalse( filtered.get(julie).contains(fill(julie, RelationType.PER_TITLE, "Singer\\/songwriter", 0.9)));
    assertTrue( filtered.get(julie).contains(fill(julie, RelationType.PER_MEMBER_OF, "Socialist Party", 1.0)));
    assertFalse( filtered.get(julie).contains(fill(julie, RelationType.PER_MEMBER_OF, "United Socialist party", 0.9)));
    assertTrue( filtered.get(julie).contains(fill(julie, RelationType.PER_TITLE, "murder defendant", 1.0)));
    assertFalse( filtered.get(julie).contains(fill(julie, RelationType.PER_TITLE, "defendant", 0.9)));
    assertTrue( filtered.get(julie).contains(fill(julie, RelationType.PER_MEMBER_OF, "American Family Association", 1.0)) );
    assertFalse( filtered.get(julie).contains(fill(julie, RelationType.PER_MEMBER_OF, "AFA", 0.9)) );

    assertTrue( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, "California", 1.0)) );
    assertFalse( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, "california", 0.9)) );
    assertTrue( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES, "Carl Â Blake", 1.0)) );
    assertFalse( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES, "Carl Blake", 0.9)) );
    assertFalse( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES, "Carl  Blake", 0.9)) );
    assertTrue( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_NUMBER_OF_EMPLOYEES_SLASH_MEMBERS, "60,000", 1.0)) );
    assertFalse( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_NUMBER_OF_EMPLOYEES_SLASH_MEMBERS, "\"60,000\"", 0.9)) );
    assertTrue( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Bernard L Madoff Investment Securities", 1.0)) );
    assertFalse( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Bernard L Madoff Investment Securities LLC", 0.9)) );
    assertTrue( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Bernard Madoff Investment Securities", 0.9)) );
    assertTrue( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Illinois Tool Works , Inc.", 1.0)) );
    assertTrue( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "Illinois Tool Works of Glenville", 0.9)) );
    assertTrue( filtered.get(stanford).contains(fill(stanford, RelationType.ORG_ALTERNATE_NAMES, "ITW", 0.8)) );
  }

  @Test
  public void testDuplicateRelationOnlyInListRelations() {
    Map<KBPEntity, List<KBPSlotFill>> filtered = new HeuristicSlotfillPostProcessors.DuplicateRelationOnlyInListRelations().postProcess(dummyData());
    assertEquals(6, filtered.get(julie).size());
    assertEquals(2, filtered.get(stanford).size());
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", 0.8)));
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "United States", 0.4)));
  }

  @Test
  public void testRespectDeclaredIncompatibilities() {
    Map<KBPEntity, List<KBPSlotFill>> filtered = new HeuristicSlotfillPostProcessors.RespectDeclaredIncompatibilities().postProcess(dummyData());
    assertEquals(8, filtered.get(julie).size());
    assertEquals(2, filtered.get(stanford).size());
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", 0.8)));
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_EMPLOYEE_OF, "Canada", 0.3)));
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.PER_ALTERNATE_NAMES, "Canada", 0.3)));
  }

  @Test
  public void testMitigateLocOfDeath() {
    Map<KBPEntity, List<KBPSlotFill>> filtered = new HeuristicSlotfillPostProcessors.MitigateLocOfDeath().postProcess(dummyData());
    assertEquals(8, filtered.get(julie).size());
    assertEquals(2, filtered.get(stanford).size());
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", 0.8)));
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_DEATH, "Switzerland", 0.5)));
  }

  @Test
  public void testCanonicalMentionRewrite() {
    Map<KBPEntity, List<KBPSlotFill>> filtered = new HeuristicSlotfillPostProcessors.CanonicalMentionRewrite().postProcess(augmentedDummyData());
    assertEquals(3, filtered.get(julie).size());
    assertEquals(2, filtered.get(kbpinc).size());
    Set<Pair<String, String>> fills = new HashSet<Pair<String, String>>();
    for (KBPSlotFill fill : filtered.get(julie)) {
      fills.add(Pair.makePair(fill.key.relationName, fill.key.slotValue));
    }
    for (KBPSlotFill fill : filtered.get(kbpinc)) {
      fills.add(Pair.makePair(fill.key.relationName, fill.key.slotValue));
    }

    assertTrue(fills.contains(Pair.makePair(RelationType.PER_COUNTRY_OF_BIRTH.canonicalName, "Canada")));
    assertTrue(fills.contains(Pair.makePair(RelationType.PER_COUNTRIES_OF_RESIDENCE.canonicalName, "Canada")));
    assertTrue(fills.contains(Pair.makePair(RelationType.PER_TITLE.canonicalName, "NLPer")));
    assertTrue(fills.contains(Pair.makePair(RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES.canonicalName, "Julie")));
    assertTrue(fills.contains(Pair.makePair(RelationType.ORG_FOUNDED.canonicalName, "2013-07-29")));
  }

  @Test
  public void testTopEmployeeRewrite() {
    Map<KBPEntity, List<KBPSlotFill>> filtered = new HeuristicSlotfillPostProcessors.TopEmployeeToFounderRewrite().postProcess(augmentedDummyData());
    assertEquals(3, filtered.get(julie).size());
    assertEquals(2, filtered.get(kbpinc).size());
    Set<Pair<String, String>> fills = new HashSet<Pair<String, String>>();
    for (KBPSlotFill fill : filtered.get(julie)) {
      fills.add(Pair.makePair(fill.key.relationName, fill.key.slotValue));
    }
    for (KBPSlotFill fill : filtered.get(kbpinc)) {
      fills.add(Pair.makePair(fill.key.relationName, fill.key.slotValue));
    }

    assertTrue(fills.contains(Pair.makePair(RelationType.PER_COUNTRY_OF_BIRTH.canonicalName, "Canada")));
    assertTrue(fills.contains(Pair.makePair(RelationType.PER_COUNTRIES_OF_RESIDENCE.canonicalName, "Canada")));
    assertTrue(fills.contains(Pair.makePair(RelationType.PER_TITLE.canonicalName, "NLPer")));
    assertTrue(fills.contains(Pair.makePair(RelationType.ORG_FOUNDED_BY.canonicalName, "she")));
    assertTrue(fills.contains(Pair.makePair(RelationType.ORG_FOUNDED.canonicalName, "July 29 2013")));
  }

  @Test
  public void testExpandMaximallyRewrite() {
    Map<KBPEntity, List<KBPSlotFill>> filtered = new HeuristicSlotfillPostProcessors.ExpandToMaximalPhraseRewrite().postProcess(augmentedDummyData());
    assertEquals(3, filtered.get(julie).size());
    assertEquals(2, filtered.get(kbpinc).size());
    Set<Pair<String, String>> fills = new HashSet<Pair<String, String>>();
    for (KBPSlotFill fill : filtered.get(julie)) {
      fills.add(Pair.makePair(fill.key.relationName, fill.key.slotValue));
    }
    for (KBPSlotFill fill : filtered.get(kbpinc)) {
      fills.add(Pair.makePair(fill.key.relationName, fill.key.slotValue));
    }

    assertTrue(fills.contains(Pair.makePair(RelationType.PER_COUNTRY_OF_BIRTH.canonicalName, "Canada")));
    assertTrue(fills.contains(Pair.makePair(RelationType.PER_COUNTRIES_OF_RESIDENCE.canonicalName, "Canada")));
    assertTrue(fills.contains(Pair.makePair(RelationType.PER_TITLE.canonicalName, "Executive NLPer")));
    assertTrue(fills.contains(Pair.makePair(RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES.canonicalName, "she")));
    assertTrue(fills.contains(Pair.makePair(RelationType.ORG_FOUNDED.canonicalName, "July 29 2013")));
  }

  @Test
  public void testExpandMaximallyRewriteRegression1() {
    Map<KBPEntity, List<KBPSlotFill>> filtered = new HeuristicSlotfillPostProcessors.ExpandToMaximalPhraseRewrite().postProcess(regression1Data());
    assertEquals(1, filtered.get(hamilton).size());
    assertTrue(filtered.get(hamilton).get(0).key.relationName.equals(RelationType.PER_TITLE.canonicalName));
    assertTrue(filtered.get(hamilton).get(0).key.slotValue.equals("pilot"));
  }

  @Test
  public void testRewriteSynergy() {
    int[][] orders = { {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    HeuristicSlotfillPostProcessor[] processors = new HeuristicSlotfillPostProcessor[] {
      new HeuristicSlotfillPostProcessors.ExpandToMaximalPhraseRewrite(),
          new HeuristicSlotfillPostProcessors.CanonicalMentionRewrite(),
          new HeuristicSlotfillPostProcessors.TopEmployeeToFounderRewrite()
    };
    for (int[] order : orders) {
      Map<KBPEntity, List<KBPSlotFill>> filtered = SlotfillPostProcessor.all(
          processors[order[0]], processors[order[1]], processors[order[2]]).postProcess(augmentedDummyData());
      assertEquals(3, filtered.get(julie).size());
      assertEquals(2, filtered.get(kbpinc).size());
      Set<Pair<String, String>> fills = new HashSet<Pair<String, String>>();
      for (KBPSlotFill fill : filtered.get(julie)) {
        fills.add(Pair.makePair(fill.key.relationName, fill.key.slotValue));
      }
      for (KBPSlotFill fill : filtered.get(kbpinc)) {
        fills.add(Pair.makePair(fill.key.relationName, fill.key.slotValue));
      }

      assertTrue(fills.contains(Pair.makePair(RelationType.PER_COUNTRY_OF_BIRTH.canonicalName, "Canada")));
      assertTrue(fills.contains(Pair.makePair(RelationType.PER_COUNTRIES_OF_RESIDENCE.canonicalName, "Canada")));
      assertTrue(fills.contains(Pair.makePair(RelationType.PER_TITLE.canonicalName, "Executive NLPer")));
      assertTrue(fills.contains(Pair.makePair(RelationType.ORG_FOUNDED_BY.canonicalName, "Julie")));
      assertTrue(fills.contains(Pair.makePair(RelationType.ORG_FOUNDED.canonicalName, "2013-07-29")));
    }

  }

  @Test
  public void testChainTwoFilters() {
    Map<KBPEntity, List<KBPSlotFill>> filtered
      = new HeuristicSlotfillPostProcessors.RespectRelationTypes().and(
        new HeuristicSlotfillPostProcessors.NoDuplicates()
    ).postProcess(dummyData());
    assertEquals(5, filtered.get(julie).size());
    assertEquals(1, filtered.get(stanford).size());
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", 0.8)));
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.ORG_MEMBERS, "Stanford CS", 0.8)));
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.PER_ALTERNATE_NAMES, "Canada", 0.8)));
  }

  @Test
  public void testChainAllFilters() {
    Map<KBPEntity, List<KBPSlotFill>> filtered = SlotfillPostProcessor.all(
        new HeuristicSlotfillPostProcessors.RespectRelationTypes(),
        new HeuristicSlotfillPostProcessors.FilterIgnoredSlots(),
        new HeuristicSlotfillPostProcessors.NoDuplicates(),
        new HeuristicSlotfillPostProcessors.DuplicateRelationOnlyInListRelations(),
        new HeuristicSlotfillPostProcessors.RespectDeclaredIncompatibilities(),
        new HeuristicSlotfillPostProcessors.MitigateLocOfDeath()
    ).postProcess(dummyData());

    assertEquals(3, filtered.get(julie).size());
    assertEquals(1, filtered.get(stanford).size());
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", 0.8)));
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_DATE_OF_BIRTH, "May 19", 0.8)));
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_EMPLOYEE_OF, "Canada", 0.4)));
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.PER_ALTERNATE_NAMES, "Canada", 0.8)));
    assertTrue(filtered.get(stanford).contains(fill(stanford, RelationType.ORG_MEMBERS, "Stanford CS", 0.8)));
  }

  @Test
  public void testURLRewriteMatcher() {
    Matcher m = HeuristicSlotfillPostProcessors.FilterUnrelatedURL.baseURL.matcher("http://www.mass.gov/legis/member/bhj1.htm");
    assertTrue(m.find());
    assertEquals("http://www.mass.gov/", m.group());
    assertEquals("mass.gov", m.group(1));

    Matcher m2 = HeuristicSlotfillPostProcessors.FilterUnrelatedURL.baseURL.matcher("http://www.bernama.com/bernama/v3/news_lite.php?id=261639");
    assertTrue(m2.find());
    assertEquals("http://www.bernama.com/", m2.group());
    assertEquals("bernama.com", m2.group(1));

    Matcher m3 = HeuristicSlotfillPostProcessors.FilterUnrelatedURL.baseURL.matcher("www.christmastree.org");
    assertTrue(m3.find());

    Matcher m4 = HeuristicSlotfillPostProcessors.FilterUnrelatedURL.baseURL.matcher("http://www.nmfa.org/");
    assertTrue(m4.find());
  }

  @Test
  public void testURLRewrite() {
    Map<KBPEntity, List<KBPSlotFill>> filtered = new HeuristicSlotfillPostProcessors.FilterUnrelatedURL().postProcess(urls());

    assertTrue(filtered.get(stanfordLong).contains(fill(stanfordLong, RelationType.ORG_WEBSITE, "http://www.stanford.edu/", 1.0)));
    assertTrue(filtered.get(stanfordLong).contains(fill(stanfordLong, RelationType.ORG_WEBSITE, "http://stanford.edu/", 1.0)));
    assertTrue(filtered.get(stanfordLong).contains(fill(stanfordLong, RelationType.ORG_WEBSITE, "http://leelandstanfordjunioruniversity.edu/", 1.0)));
    assertTrue(filtered.get(stanfordLong).contains(fill(stanfordLong, RelationType.ORG_WEBSITE, "http://leelandstanforduniversity.edu/", 1.0)));
    assertTrue(filtered.get(stanfordLong).contains(fill(stanfordLong, RelationType.ORG_WEBSITE, "http://lsju.edu/", 1.0)));
    assertTrue(filtered.get(stanfordLong).contains(fill(stanfordLong, RelationType.ORG_WEBSITE, "http://stanford.co.uk/", 1.0)));

    assertFalse(filtered.get(stanfordLong).contains(fill(stanfordLong, RelationType.ORG_WEBSITE, "http://www.foo.edu/", 1.0)));
    assertFalse(filtered.get(stanfordLong).contains(fill(stanfordLong, RelationType.ORG_WEBSITE, "http://www.bar.edu/stanford", 1.0)));
  }

  @Test
  public void testBornInRewrite() {
    Map<KBPEntity, List<KBPSlotFill>> filtered = new HeuristicSlotfillPostProcessors.BornInRewrite().postProcess(births());
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Canada", 0.2)));
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_CITY_OF_BIRTH, "San Francisco", 0.4))); // technically wrong
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH, "Montana", 0.4)));  // technically wrong
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_CITIES_OF_RESIDENCE, "Toronto", 0.4)));
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.PER_COUNTRY_OF_BIRTH, "Mexico", 0.2)));
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_DATE_OF_BIRTH, "December", 0.2)));
    assertEquals(5, filtered.get(julie).size());
  }

  @Test
  public void testConformToGuidelinesFilter() {
    Map<KBPEntity, List<KBPSlotFill>> data = new HashMap<KBPEntity, List<KBPSlotFill>>();
    data.put(julie, new ArrayList<KBPSlotFill>());  // note: most of these don't actually apply to the real Julie
    data.get(julie).add(fill(julie, RelationType.PER_DATE_OF_BIRTH, "1990", 1.0));
    data.get(julie).add(fill(julie, RelationType.PER_TITLE, "hero", 1.0));
    data.put(stanford, new ArrayList<KBPSlotFill>());  // note: most of these don't actually apply to the real Julie
    data.get(stanford).add(fill(stanford, RelationType.ORG_FOUNDED, "Friday", 1.0));
    data.get(stanford).add(fill(stanford, RelationType.ORG_DISSOLVED, "1885-11", 1.0));

    Map<KBPEntity, List<KBPSlotFill>> filtered = new HeuristicSlotfillPostProcessors.ConformToGuidelinesFilter().postProcess(data);
    assertTrue(filtered.get(julie).contains(fill(julie, RelationType.PER_DATE_OF_BIRTH, "1990-XX-XX", 1.0)));
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.PER_TITLE, "spokeswoman", 1.0)));
    assertTrue(filtered.get(stanford).contains(fill(stanford, RelationType.ORG_DISSOLVED, "1885-11-XX", 1.0)));
    assertFalse(filtered.get(stanford).contains(fill(stanford, RelationType.ORG_FOUNDED, "Friday", 1.0)));

    // Test date normalization
    filtered = new HeuristicSlotfillPostProcessors.ConformToGuidelinesFilter().postProcess(births());
    assertFalse(filtered.get(julie).contains(fill(julie, RelationType.PER_DATE_OF_BIRTH, "December", 0.2)));
    assertFalse(filtered.get(hamilton).contains(fill(hamilton, RelationType.PER_DATE_OF_DEATH, "last month", 0.8)));


  }

}
