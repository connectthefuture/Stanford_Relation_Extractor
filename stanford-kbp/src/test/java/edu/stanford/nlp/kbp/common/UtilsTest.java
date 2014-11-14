package edu.stanford.nlp.kbp.common;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import java.util.function.Function;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.*;

/**
 * Unit test some utilities.
 * Corollary: try to make utilities unit testable?
 *
 * @author Gabor Angeli
 */
public class UtilsTest {

  @Test
  public void testGetTokenSpanSimple() {
    char[][] haystack = new char[][]{"hello".toCharArray(), "world".toCharArray()};
    char[] needle = "hello".toCharArray();
    assertEquals(new Span(0, 1), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
    needle = "world".toCharArray();
    assertEquals(new Span(1, 2), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
    needle = "helloworld".toCharArray();
    assertEquals(new Span(0, 2), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
  }

  @Test
  public void testGetTokenSpanWhitespaceSimple() {
    char[][] haystack = new char[][]{"hello".toCharArray(), "world".toCharArray()};
    char[] needle = "hello world".toCharArray();
    assertEquals(new Span(0, 2), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
    needle = "  hello".toCharArray();
    assertEquals(new Span(0, 1), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
    needle = "  hello    world  ".toCharArray();
    assertEquals(new Span(0, 2), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
  }

  @Test
  public void testGetTokenSpanSplitToken() {
    char[][] haystack = new char[][]{"hel".toCharArray(), "lo".toCharArray(), "world".toCharArray()};
    char[] needle = "hello world".toCharArray();
    assertEquals(new Span(0, 3), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
    needle = "  hello".toCharArray();
    assertEquals(new Span(0, 2), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
    needle = "  hello    world  ".toCharArray();
    assertEquals(new Span(0, 3), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
  }

  @Test
  public void testGetTokenMisalignedBoundaries() {
    char[][] haystack = new char[][]{"hel".toCharArray(), "lo".toCharArray(), "world".toCharArray()};
    char[] needle = "llo world".toCharArray();
    assertEquals(new Span(0, 3), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
    needle = "  hel".toCharArray();
    assertEquals(new Span(0, 1), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
    needle = "  hello    wor  ".toCharArray();
    assertEquals(new Span(0, 3), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
    needle = "hello w".toCharArray();
    assertEquals(new Span(0, 3), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
  }

  @Test
  public void testGetTokenSpanWhitespaceComplex() {
    char[][] haystack = new char[][]{"hel  lo".toCharArray(), "w o rld  ".toCharArray()};
    char[] needle = "hello world".toCharArray();
    assertEquals(new Span(0, 2), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
    needle = "  hello".toCharArray();
    assertEquals(new Span(0, 1), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
    needle = "  hello    world  ".toCharArray();
    assertEquals(new Span(0, 2), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
  }

  @Test
  public void testGetTokenSpanComplexRegexBehavior() {
    char[][] haystack = new char[][]{"aba".toCharArray(), "ba".toCharArray(), "c".toCharArray()};
    char[] needle = "babc".toCharArray();
    assertEquals(new Span(0, 3), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
    needle = "  ba  bc ".toCharArray();
    assertEquals(new Span(0, 3), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
  }

  @Test
  public void testGetTokenSpanMultipleMatchBehavior() {
    char[][] haystack = new char[][]{"the".toCharArray(), "cat".toCharArray(), "the".toCharArray(), "dog".toCharArray(),
        "the".toCharArray(), "cat".toCharArray()};
    char[] needle = "the cat".toCharArray();
    assertEquals(new Span(0, 2), Utils.getTokenSpan(haystack, needle, Maybe.<Span>Nothing()).getOrElse(new Span(-1, 0)));
    needle = "the cat".toCharArray();
    assertEquals(new Span(4, 6), Utils.getTokenSpan(haystack, needle, Maybe.Just(new Span(4, 6))).getOrElse(new Span(-1, 0)));
    needle = "the cat".toCharArray();
    assertEquals(new Span(4, 6), Utils.getTokenSpan(haystack, needle, Maybe.Just(new Span(3, 5))).getOrElse(new Span(-1, 0)));
    needle = "the cat".toCharArray();
    assertEquals(new Span(0, 2), Utils.getTokenSpan(haystack, needle, Maybe.Just(new Span(0, 4))).getOrElse(new Span(-1, 0)));
  }

  @Test
  public void testSortRelationsByPrior() {
    Set<String> relations = new HashSet<String>() {{
      add(RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH.canonicalName);
      add(RelationType.PER_RELIGION.canonicalName);
      add(RelationType.PER_ALTERNATE_NAMES.canonicalName);
    }};
    List<String> sortedRelations = Utils.sortRelationsByPrior(relations);
    assertEquals(RelationType.PER_ALTERNATE_NAMES.canonicalName, sortedRelations.get(0));
    assertEquals(RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH.canonicalName, sortedRelations.get(1));
    assertEquals(RelationType.PER_RELIGION.canonicalName, sortedRelations.get(2));
  }

  @Test
  public void testSortRelationsByPriorUnknownRelations() {
    Set<String> relations = new HashSet<String>() {{
      add(RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH.canonicalName);
      add(RelationType.PER_RELIGION.canonicalName);
      add(RelationType.PER_ALTERNATE_NAMES.canonicalName);
      add("foo");
      add("bar");
      add("baz");
      add("bam");
    }};
    List<String> sortedRelations = Utils.sortRelationsByPrior(relations);
    assertEquals(RelationType.PER_ALTERNATE_NAMES.canonicalName, sortedRelations.get(0));
    assertEquals(RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH.canonicalName, sortedRelations.get(1));
    assertEquals(RelationType.PER_RELIGION.canonicalName, sortedRelations.get(2));
    assertEquals("bam", sortedRelations.get(3));
    assertEquals("bar", sortedRelations.get(4));
    assertEquals("baz", sortedRelations.get(5));
    assertEquals("foo", sortedRelations.get(6));
  }

  @Test
  public void testIsLoop() {
    // Test some true loops
    List<KBPSlotFill> loopA = new ArrayList<KBPSlotFill>() {{
      add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("ORGANIZATION").rel("rAB").KBPSlotFill());
      add(KBPNew.entName("B").entType("ORGANIZATION").slotValue("A").slotType("PERSON").rel("rBA").KBPSlotFill());
    }};
    assertTrue(Utils.isLoopPath(loopA));

    List<KBPSlotFill> loopB = new ArrayList<KBPSlotFill>() {{
      add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("ORGANIZATION").rel("rAB").KBPSlotFill());
      add(KBPNew.entName("B").entType("ORGANIZATION").slotValue("C").slotType("PERSON").rel("rBC").KBPSlotFill());
      add(KBPNew.entName("C").entType("PERSON").slotValue("A").slotType("PERSON").rel("rCA").KBPSlotFill());
    }};
    assertTrue(Utils.isLoopPath(loopB));

    List<KBPSlotFill> loopReverse = new ArrayList<KBPSlotFill>() {{
      add(KBPNew.entName("B").entType("ORGANIZATION").slotValue("A").slotType("PERSON").rel("rBA").KBPSlotFill());
      add(KBPNew.entName("B").entType("ORGANIZATION").slotValue("C").slotType("PERSON").rel("rBC").KBPSlotFill());
      add(KBPNew.entName("A").entType("PERSON").slotValue("C").slotType("PERSON").rel("rAC").KBPSlotFill());
    }};
    assertTrue(Utils.isLoopPath(loopReverse));

    List<KBPSlotFill> loopReverse2 = new ArrayList<KBPSlotFill>() {{
      add(KBPNew.entName("B").entType("ORGANIZATION").slotValue("A").slotType("PERSON").rel("rBA").KBPSlotFill());
      add(KBPNew.entName("B").entType("ORGANIZATION").slotValue("C").slotType("PERSON").rel("rBC").KBPSlotFill());
      add(KBPNew.entName("C").entType("PERSON").slotValue("A").slotType("PERSON").rel("rCA").KBPSlotFill());
    }};
    assertTrue(Utils.isLoopPath(loopReverse2));

    // Test some false loops
    List<KBPSlotFill> unLoopA = new ArrayList<KBPSlotFill>() {{
      add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("ORGANIZATION").rel("rAB").KBPSlotFill());
      add(KBPNew.entName("B").entType("ORGANIZATION").slotValue("A").slotType("ORGANIZATION").rel("rBA").KBPSlotFill());
    }};
    assertFalse(Utils.isLoopPath(unLoopA));

    List<KBPSlotFill> unLoopB = new ArrayList<KBPSlotFill>() {{
      add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("ORGANIZATION").rel("rAB").KBPSlotFill());
      add(KBPNew.entName("B").entType("ORGANIZATION").slotValue("C").slotType("ORGANIZATION").rel("rBC").KBPSlotFill());
      add(KBPNew.entName("C").entType("ORGANIZATION").slotValue("A").slotType("PERSON").rel("rCA").KBPSlotFill());
      add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("ORGANIZATION").rel("rAB").KBPSlotFill());
    }};
    assertFalse(Utils.isLoopPath(unLoopB));

    List<KBPSlotFill> unLoopC = new ArrayList<KBPSlotFill>() {{
      add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("ORGANIZATION").rel("rAB").KBPSlotFill());
      add(KBPNew.entName("B").entType("ORGANIZATION").slotValue("C").slotType("ORGANIZATION").rel("rBC").KBPSlotFill());
      add(KBPNew.entName("C").entType("ORGANIZATION").slotValue("A").slotType("PERSON").rel("rCA").KBPSlotFill());
      add(KBPNew.entName("A").entType("PERSON").slotValue("D").slotType("ORGANIZATION").rel("rAD").KBPSlotFill());
    }};
    assertFalse(Utils.isLoopPath(unLoopC));

    List<KBPSlotFill> unLoopD = new ArrayList<KBPSlotFill>() {{
      add(KBPNew.entName("Canada").entType("COUNTRY").slotValue("Gabor").slotType("PERSON").rel("founded_by").KBPSlotFill());
      add(KBPNew.entName("Julie").entType("PERSON").slotValue("Canada").slotType("COUNTRY").rel("country_of_birth").KBPSlotFill());
    }};
    assertFalse(Utils.isLoopPath(unLoopD));
  }

  @Test
  public void testIsLoopRegressions() {
    // A weird case of A->B->C->A<-D that showed up
    List<KBPSlotFill> loopFull = new ArrayList<KBPSlotFill>() {{
      add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("PERSON").rel("rAB").KBPSlotFill());
      add(KBPNew.entName("B").entType("PERSON").slotValue("C").slotType("PERSON").rel("rBC").KBPSlotFill());
      add(KBPNew.entName("C").entType("PERSON").slotValue("D").slotType("PERSON").rel("rCD").KBPSlotFill());
      add(KBPNew.entName("B").entType("PERSON").slotValue("D").slotType("PERSON").rel("rBD").KBPSlotFill());
    }};
    assertFalse(Utils.isLoopPath(loopFull));
    List<KBPSlotFill> loopPart = new ArrayList<KBPSlotFill>() {{
      add(KBPNew.entName("B").entType("PERSON").slotValue("C").slotType("PERSON").rel("rBC").KBPSlotFill());
      add(KBPNew.entName("C").entType("PERSON").slotValue("D").slotType("PERSON").rel("rCD").KBPSlotFill());
      add(KBPNew.entName("B").entType("PERSON").slotValue("D").slotType("PERSON").rel("rBD").KBPSlotFill());
    }};
    assertTrue(Utils.isLoopPath(loopPart));

    // Technically, cloning the same relation (though valid) should be a loop
    List<KBPSlotFill> loopDuplicateArch = new ArrayList<KBPSlotFill>() {{
      add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("PERSON").rel("rAB").KBPSlotFill());
      add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("PERSON").rel("rAB2").KBPSlotFill());
    }};
    assertTrue(Utils.isLoopPath(loopDuplicateArch));
    List<KBPSlotFill> loopDuplicateArch2 = new ArrayList<KBPSlotFill>() {{
      add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("PERSON").rel("rAB").KBPSlotFill());
      add(KBPNew.entName("B").entType("PERSON").slotValue("A").slotType("PERSON").rel("rAB2").KBPSlotFill());
    }};
    assertTrue(Utils.isLoopPath(loopDuplicateArch2));

    // Another weird case -- maybe this was the bug?
    List<KBPSlotFill> loopX = new ArrayList<KBPSlotFill>() {{
      add(KBPNew.entName("C").entType("PERSON").slotValue("B").slotType("PERSON").rel("r1").KBPSlotFill());
      add(KBPNew.entName("B").entType("PERSON").slotValue("C").slotType("PERSON").rel("r2").KBPSlotFill());
      add(KBPNew.entName("C").entType("PERSON").slotValue("B").slotType("PERSON").rel("r3").KBPSlotFill());
      add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("PERSON").rel("r4").KBPSlotFill());
    }};
    assertFalse(Utils.isLoopPath(loopX));
    List<KBPSlotFill> loopY = new ArrayList<KBPSlotFill>() {{
      add(KBPNew.entName("C").entType("PERSON").slotValue("B").slotType("PERSON").rel("r1").KBPSlotFill());
      add(KBPNew.entName("B").entType("PERSON").slotValue("C").slotType("PERSON").rel("r2").KBPSlotFill());
    }};
    assertTrue(Utils.isLoopPath(loopY));
  }

  @Test
  public void testAntecedents() {
    Set<KBTriple> loop = new HashSet<KBTriple>() {{
      add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("PERSON").rel("aaa").KBTriple());
      add(KBPNew.entName("B").entType("PERSON").slotValue("C").slotType("PERSON").rel("bbb").KBTriple());
      add(KBPNew.entName("C").entType("PERSON").slotValue("D").slotType("PERSON").rel("ccc").KBTriple());
      add(KBPNew.entName("D").entType("PERSON").slotValue("A").slotType("PERSON").rel("ddd").KBTriple());
    }};
    assertEquals(new HashSet<Set<KBTriple>>() {{
      add(new HashSet<KBTriple>() {{
        add(KBPNew.entName("B").entType("PERSON").slotValue("C").slotType("PERSON").rel("bbb").KBTriple());
        add(KBPNew.entName("C").entType("PERSON").slotValue("D").slotType("PERSON").rel("ccc").KBTriple());
        add(KBPNew.entName("D").entType("PERSON").slotValue("A").slotType("PERSON").rel("ddd").KBTriple());
      }});
      add(new HashSet<KBTriple>() {{
        add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("PERSON").rel("aaa").KBTriple());
        add(KBPNew.entName("C").entType("PERSON").slotValue("D").slotType("PERSON").rel("ccc").KBTriple());
        add(KBPNew.entName("D").entType("PERSON").slotValue("A").slotType("PERSON").rel("ddd").KBTriple());
      }});
      add(new HashSet<KBTriple>() {{
        add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("PERSON").rel("aaa").KBTriple());
        add(KBPNew.entName("B").entType("PERSON").slotValue("C").slotType("PERSON").rel("bbb").KBTriple());
        add(KBPNew.entName("D").entType("PERSON").slotValue("A").slotType("PERSON").rel("ddd").KBTriple());
      }});
      add(new HashSet<KBTriple>() {{
        add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("PERSON").rel("aaa").KBTriple());
        add(KBPNew.entName("B").entType("PERSON").slotValue("C").slotType("PERSON").rel("bbb").KBTriple());
        add(KBPNew.entName("C").entType("PERSON").slotValue("D").slotType("PERSON").rel("ccc").KBTriple());
      }});
    }}, Utils.getValidAntecedents(loop));

    loop = new HashSet<KBTriple>() {{
      add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("PERSON").rel("aaa").KBTriple());
      add(KBPNew.entName("C").entType("PERSON").slotValue("B").slotType("PERSON").rel("bbb").KBTriple());
      add(KBPNew.entName("C").entType("PERSON").slotValue("D").slotType("PERSON").rel("ccc").KBTriple());
      add(KBPNew.entName("A").entType("PERSON").slotValue("D").slotType("PERSON").rel("ddd").KBTriple());
    }};
    assertEquals(new HashSet<Set<KBTriple>>() {{
      add(new HashSet<KBTriple>() {{
        add(KBPNew.entName("C").entType("PERSON").slotValue("B").slotType("PERSON").rel("bbb").KBTriple());
        add(KBPNew.entName("C").entType("PERSON").slotValue("D").slotType("PERSON").rel("ccc").KBTriple());
        add(KBPNew.entName("A").entType("PERSON").slotValue("D").slotType("PERSON").rel("ddd").KBTriple());
      }});
      add(new HashSet<KBTriple>() {{
        add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("PERSON").rel("aaa").KBTriple());
        add(KBPNew.entName("C").entType("PERSON").slotValue("D").slotType("PERSON").rel("ccc").KBTriple());
        add(KBPNew.entName("A").entType("PERSON").slotValue("D").slotType("PERSON").rel("ddd").KBTriple());
      }});
      add(new HashSet<KBTriple>() {{
        add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("PERSON").rel("aaa").KBTriple());
        add(KBPNew.entName("C").entType("PERSON").slotValue("B").slotType("PERSON").rel("bbb").KBTriple());
        add(KBPNew.entName("A").entType("PERSON").slotValue("D").slotType("PERSON").rel("ddd").KBTriple());
      }});
      add(new HashSet<KBTriple>() {{
        add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("PERSON").rel("aaa").KBTriple());
        add(KBPNew.entName("C").entType("PERSON").slotValue("B").slotType("PERSON").rel("bbb").KBTriple());
        add(KBPNew.entName("C").entType("PERSON").slotValue("D").slotType("PERSON").rel("ccc").KBTriple());
      }});
    }}, Utils.getValidAntecedents(loop));
  }

  @Test
  public void testNormalizeClause() {
    // Make sure we abstract entities properly
    Set<KBPSlotFill> loop = new HashSet<KBPSlotFill>() {{
      add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("PERSON").rel("bbbRel").KBPSlotFill());
      add(KBPNew.entName("B").entType("PERSON").slotValue("A").slotType("PERSON").rel("aaaRel").KBPSlotFill());
    }};
    assertEquals(new HashSet<KBTriple>() {{
      add(KBPNew.entName("x1").entType("PERSON").slotValue("x0").slotType("PERSON").rel("bbbRel").KBTriple());
      add(KBPNew.entName("x0").entType("PERSON").slotValue("x1").slotType("PERSON").rel("aaaRel").KBTriple());
    }}, Utils.abstractConjunction(loop).second);

    // Abstract one of the clauses
    assertEquals(new HashSet<KBTriple>() {{
      add(KBPNew.entName("x0").entType("PERSON").slotValue("x1").slotType("PERSON").rel("bbbRel").KBTriple());
    }}, Utils.abstractConjunction(new HashSet<KBPSlotFill>() {{
      add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("PERSON").rel("bbbRel").KBPSlotFill());
    }}).second);

    // Abstract the other clause
    assertEquals(new HashSet<KBTriple>() {{
      add(KBPNew.entName("x0").entType("PERSON").slotValue("x1").slotType("PERSON").rel("aaaRel").KBTriple());
    }}, Utils.abstractConjunction(new HashSet<KBPSlotFill>() {{
      add(KBPNew.entName("B").entType("PERSON").slotValue("A").slotType("PERSON").rel("aaaRel").KBPSlotFill());
    }}).second);

    // The antecedents should look exactly like the abstracted clauses.
    // Note that this is different from what they looked like in the original antecedent generation
    assertEquals(new HashSet<Set<KBTriple>>() {{
      add(new HashSet<KBTriple>() {{
        add(KBPNew.entName("x0").entType("PERSON").slotValue("x1").slotType("PERSON").rel("aaaRel").KBTriple());
      }});
      add(new HashSet<KBTriple>() {{
        add(KBPNew.entName("x0").entType("PERSON").slotValue("x1").slotType("PERSON").rel("bbbRel").KBTriple());
      }});
    }}, Utils.getValidNormalizedAntecedents(CollectionUtils.lazyMap(loop, new Function<KBPSlotFill, KBTriple>() { @Override public KBTriple apply(KBPSlotFill in) { return in.key; } } )));
    for (Set<KBTriple> entry : Utils.getValidNormalizedAntecedents(CollectionUtils.lazyMap(loop, new Function<KBPSlotFill, KBTriple>() { @Override public KBTriple apply(KBPSlotFill in) { return in.key; } } ))) {
      assertEquals(entry, Utils.normalizeConjunction(entry));
    }

  }

  private void ensureConsistent(List<KBTriple> formula) {
    Set<Set<KBTriple>> normalizedVersions = new HashSet<>();
    for (List<KBTriple> permutation : CollectionUtils.permutations(formula)) {
      normalizedVersions.add(Utils.normalizeConjunction(permutation));
      assertEquals(Utils.normalizeConjunction(permutation), Utils.normalizeConjunction(Utils.normalizeConjunction(permutation)));
    }
    assertEquals(1, normalizedVersions.size());
    assertEquals(normalizedVersions.iterator().next(), Utils.normalizeConjunction(normalizedVersions.iterator().next()));
  }

  @Test
  public void testNormalizeClauseOrder() {
    // Handle new variable order
    ensureConsistent( new ArrayList<KBTriple>() {{
      add(KBPNew.entName("x0").entType("PERSON").slotValue("x1").slotType("PERSON").rel("coh").KBTriple());
      add(KBPNew.entName("x0").entType("PERSON").slotValue("x3").slotType("PERSON").rel("soh").KBTriple());
      add(KBPNew.entName("x2").entType("PERSON").slotValue("x3").slotType("PERSON").rel("soh").KBTriple());
    }});
    // Handle number of clumped clauses order
    ensureConsistent(new ArrayList<KBTriple>() {{
      add(KBPNew.entName("x0").entType("PERSON").slotValue("x1").slotType("PERSON").rel("leave").KBTriple());
      add(KBPNew.entName("x2").entType("PERSON").slotValue("x1").slotType("PERSON").rel("leave").KBTriple());
      add(KBPNew.entName("x3").entType("PERSON").slotValue("x0").slotType("PERSON").rel("stay_from").KBTriple());
    }});
    // Handle ties
    ensureConsistent(new ArrayList<KBTriple>() {{
      add(KBPNew.entName("x0").entType("PERSON").slotValue("x1").slotType("PERSON").rel("altname").KBTriple());
      add(KBPNew.entName("x1").entType("PERSON").slotValue("x2").slotType("PERSON").rel("employee").KBTriple());
      add(KBPNew.entName("x0").entType("PERSON").slotValue("x3").slotType("PERSON").rel("employee").KBTriple());
    }});
    // Handle more ties!
    ensureConsistent(new ArrayList<KBTriple>() {{
      add(KBPNew.entName("x0").entType("PERSON").slotValue("x1").slotType("PERSON").rel("altname").KBTriple());
      add(KBPNew.entName("x0").entType("PERSON").slotValue("x2").slotType("MISC").rel("altname").KBTriple());
      add(KBPNew.entName("x3").entType("PERSON").slotValue("x1").slotType("PERSON").rel("altname").KBTriple());
    }});
    // Really nasty cases
    ensureConsistent(new ArrayList<KBTriple>() {{
      add(KBPNew.entName("x0").entType("PERSON").slotValue("x1").slotType("PERSON").rel("r").KBTriple());
      add(KBPNew.entName("x2").entType("PERSON").slotValue("x0").slotType("PERSON").rel("r").KBTriple());
    }});
    ensureConsistent(new ArrayList<KBTriple>() {{
      add(KBPNew.entName("foo").entType("PERSON").slotValue("bar").slotType("PERSON").rel("call").KBTriple());
      add(KBPNew.entName("baz").entType("PERSON").slotValue("yam").slotType("MISC").rel("per:alternate_names").KBTriple());
      add(KBPNew.entName("foo").entType("PERSON").slotValue("yam").slotType("MISC").rel("per:alternate_names").KBTriple());
    }});
  }

  @Test
  public void testNormalizeOrderedConjunction() {
    // Test order in one direction
    List<KBTriple> loop = new ArrayList<KBTriple>() {{
      add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("PERSON").rel("bbbRel").KBTriple());
      add(KBPNew.entName("B").entType("PERSON").slotValue("A").slotType("PERSON").rel("aaaRel").KBTriple());
    }};
    assertEquals(new ArrayList<KBTriple>() {{
      add(KBPNew.entName("x0").entType("PERSON").slotValue("x1").slotType("PERSON").rel("bbbRel").KBTriple());
      add(KBPNew.entName("x1").entType("PERSON").slotValue("x0").slotType("PERSON").rel("aaaRel").KBTriple());
    }}, Utils.normalizeOrderedConjunction(loop));

    // Test order in the other direction
    loop = new ArrayList<KBTriple>() {{
      add(KBPNew.entName("B").entType("PERSON").slotValue("A").slotType("PERSON").rel("aaaRel").KBTriple());
      add(KBPNew.entName("A").entType("PERSON").slotValue("B").slotType("PERSON").rel("bbbRel").KBTriple());
    }};
    assertEquals(new ArrayList<KBTriple>() {{
      add(KBPNew.entName("x0").entType("PERSON").slotValue("x1").slotType("PERSON").rel("aaaRel").KBTriple());
      add(KBPNew.entName("x1").entType("PERSON").slotValue("x0").slotType("PERSON").rel("bbbRel").KBTriple());
    }}, Utils.normalizeOrderedConjunction(loop));
  }

  @Test
  public void testNoSpecialCharacters() {
    assertEquals("foo", Utils.noSpecialChars("foo"));
    assertEquals("foobar", Utils.noSpecialChars("foo\\bar"));
    assertEquals("foobar", Utils.noSpecialChars("foo-bar"));
    assertEquals("foobar", Utils.noSpecialChars("foo\"bar"));
    assertEquals("foobar", Utils.noSpecialChars("Foo\"bAr"));

  }
}
