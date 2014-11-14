package edu.stanford.nlp.kbp.slotfilling.scripts;

import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.scripts.MineInferentialPaths.Trie;
import static edu.stanford.nlp.kbp.slotfilling.scripts.MineInferentialPaths.Direction.*;
import edu.stanford.nlp.util.Triple;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import static junit.framework.Assert.*;

/**
 * A unit test for the Trie used in MineInferentialPaths
 *
 * @author Gabor Angeli
 */
public class MineInferentialPathsTrieTest {
  private KBPEntity julie = KBPNew.entName("Julie").entType(NERTag.PERSON).KBPEntity();
  private KBPEntity arun = KBPNew.entName("Arun").entType(NERTag.PERSON).KBPEntity();
  private KBPEntity gabor = KBPNew.entName("Gabor").entType(NERTag.PERSON).KBPEntity();
  private KBPEntity chris = KBPNew.entName("Chris").entType(NERTag.PERSON).KBPEntity();
  private KBPEntity percy = KBPNew.entName("Percy").entType(NERTag.PERSON).KBPEntity();
  private KBPEntity stanford = KBPNew.entName("Stanford").entType(NERTag.ORGANIZATION).KBPEntity();
  private KBPEntity canada = KBPNew.entName("Canada").entType(NERTag.COUNTRY).KBPEntity();

  @Test
  public void testConstructors() {
    Trie t = new Trie(julie);
    assertNull(t.parent);
    assertNull(t.relationFromParent);
    assertNotNull(t.children);
    assertEquals(julie, t.entry);

    Trie u = new Trie(Triple.makeTriple("r", FORWARD, canada), t);
    assertEquals(t, u.parent);
    assertNotNull(u.relationFromParent);
    assertEquals("r", u.relationFromParent.first);
    assertEquals(FORWARD, u.relationFromParent.second);
    assertNotNull(u.children);
    assertEquals(canada, u.entry);
  }

  @Test
  public void testDepth() {
    Trie t = new Trie(julie);
    Trie u = new Trie(Triple.makeTriple("r", FORWARD, canada), t);
    Trie v = new Trie(Triple.makeTriple("r", BACKWARD, arun), u);
    assertEquals(0, t.depth());
    assertEquals(1, u.depth());
    assertEquals(2, v.depth());
  }

  @Test
  public void testRoot() {
    Trie t = new Trie(julie);
    Trie u = new Trie(Triple.makeTriple("r", FORWARD, canada), t);
    Trie v = new Trie(Triple.makeTriple("r", BACKWARD, arun), u);
    assertEquals(julie, t.root());
    assertEquals(julie, u.root());
    assertEquals(julie, v.root());
  }

  @Test
  public void testIsLoop() {
    Trie t = new Trie(julie);
    Trie u = new Trie(Triple.makeTriple("r", FORWARD, canada), t);
    Trie v = new Trie(Triple.makeTriple("r", BACKWARD, arun), u);
    Trie l = new Trie(Triple.makeTriple("r", FORWARD, julie), u);
    assertFalse(t.isLoop());
    assertFalse(u.isLoop());
    assertFalse(v.isLoop());
    assertTrue(l.isLoop());
  }

  @Test
  public void testDanglingLoop() {
    Trie t = new Trie(julie);
    Trie u = new Trie(Triple.makeTriple("r", FORWARD, canada), t);
    Trie v = new Trie(Triple.makeTriple("r", BACKWARD, arun), u);

    assertFalse(t.danglingLoop(canada));
    assertFalse(t.danglingLoop(julie));  // reflexive relation

    assertFalse(u.danglingLoop(arun));
    assertFalse(u.danglingLoop(julie));  // full loop

    assertFalse(v.danglingLoop(chris));
    assertFalse(v.danglingLoop(julie));  // full loop
    assertTrue(v.danglingLoop(canada));  // !! dangling loop !!
  }

  @Test
  public void testExtendSimpleCase() {
    Trie t = new Trie(julie);
    Trie u = t.extend(KBPNew.from(julie).slotValue(canada).rel("r").KBTriple()).orCrash();
    assertEquals(new Trie(Triple.makeTriple("r", FORWARD, canada), t), u);
    assertFalse(t.children.isEmpty());
    Trie v = u.extend(KBPNew.from(arun).slotValue(canada).rel("r").KBTriple()).orCrash();
    assertEquals(new Trie(Triple.makeTriple("r", BACKWARD, arun), u), v);
    assertFalse(u.children.isEmpty());
  }

  @Test
  public void testExtendDisallowDanglingLoops() {
    Trie t = new Trie(julie);
    Trie u = t.extend(KBPNew.from(julie).slotValue(canada).rel("r").KBTriple()).orCrash();
    Trie v = u.extend(KBPNew.from(arun).slotValue(canada).rel("r").KBTriple()).orCrash();

    assertFalse(v.extend(KBPNew.from(arun).slotValue(canada).rel("r").KBTriple()).isDefined());
    assertTrue(v.children.isEmpty());

    assertFalse(v.extend(KBPNew.from(canada).slotValue(arun).rel("r").KBTriple()).isDefined());
    assertTrue(v.children.isEmpty());
  }

  @Test
  public void testExtendDisallowBacktracking() {
    Trie t = new Trie(julie);
    Trie u = t.extend(KBPNew.from(julie).slotValue(canada).rel("r").KBTriple()).orCrash();

    assertFalse(u.extend(KBPNew.from(julie).slotValue(canada).rel("r").KBTriple()).isDefined());
    assertTrue(u.children.isEmpty());

    assertFalse(u.extend(KBPNew.from(canada).slotValue(julie).rel("s").KBTriple()).isDefined());  // it's a loop
    assertFalse(u.children.isEmpty());

    assertFalse(u.extend(KBPNew.from(julie).slotValue(canada).rel("s").KBTriple()).isDefined());  // it's a loop
    assertEquals(2, u.children.size());
  }

  @Test
  public void testExtendAllowReflexiveLoops() {
    Trie t = new Trie(julie);
    Trie u = t.extend(KBPNew.from(julie).slotValue(canada).rel("r").KBTriple()).orCrash();

    assertFalse(u.extend(KBPNew.from(canada).slotValue(julie).rel("r").KBTriple()).isDefined());  // it's a loop
    assertFalse(u.children.isEmpty());
  }

  @Test
  public void testExtendAllowButDontExtendClosedLoops() {
    Trie t = new Trie(julie);
    Trie u = t.extend(KBPNew.from(julie).slotValue(canada).rel("r").KBTriple()).orCrash();
    Trie v = u.extend(KBPNew.from(arun).slotValue(canada).rel("r").KBTriple()).orCrash();

    assertFalse(v.extend(KBPNew.from(arun).slotValue(julie).rel("r").KBTriple()).isDefined());
    assertFalse(v.children.isEmpty());

    assertFalse(v.extend(KBPNew.from(julie).slotValue(arun).rel("r").KBTriple()).isDefined());
    assertFalse(v.children.isEmpty());
  }

  @Test
  public void testExtendMaxDepth() {
    int savedDepth = Props.TEST_GRAPH_INFERENCE_DEPTH;
    Props.TEST_GRAPH_INFERENCE_DEPTH = 3;
    Trie t = new Trie(julie);
    Trie u = t.extend(KBPNew.from(julie).slotValue(canada).rel("r").KBTriple()).orCrash();
    Trie v = u.extend(KBPNew.from(arun).slotValue(canada).rel("r").KBTriple()).orCrash();
    Trie w = v.extend(KBPNew.from(arun).slotValue(gabor).rel("r").KBTriple()).orCrash();
    assertFalse(v.children.isEmpty());
    assertFalse(w.extend(KBPNew.from(gabor).slotValue(stanford).rel("r").KBTriple()).isDefined());
    assertTrue(w.children.isEmpty());
    Props.TEST_GRAPH_INFERENCE_DEPTH = savedDepth;
  }

  @Test
  public void testExtendDuplicatePath() {
    Trie t = new Trie(julie);
    Trie u = t.extend(KBPNew.from(julie).slotValue(canada).rel("r").KBTriple()).orCrash();
    u.extend(KBPNew.from(arun).slotValue(canada).rel("r").KBTriple()).orCrash();
    assertEquals(1, u.children.size());
    assertFalse(u.extend(KBPNew.from(arun).slotValue(canada).rel("r").KBTriple()).isDefined());
    assertEquals(1, u.children.size());
  }

  @Test
  public void testAsPath() {
    int savedDepth = Props.TEST_GRAPH_INFERENCE_DEPTH;
    Props.TEST_GRAPH_INFERENCE_DEPTH = 4;
    Trie t = new Trie(julie);
    Trie u = t.extend(KBPNew.from(julie).slotValue(canada).rel("r").KBTriple()).orCrash();
    Trie v = u.extend(KBPNew.from(arun).slotValue(canada).rel("r").KBTriple()).orCrash();
    Trie w = v.extend(KBPNew.from(arun).slotValue(gabor).rel("r").KBTriple()).orCrash();

    assertEquals(
        new ArrayList<KBTriple>(){{
          add(KBPNew.from(julie).slotValue(canada).rel("r").KBTriple());
        }},
        u.asPath());
    assertEquals(
        new ArrayList<KBTriple>(){{
          add(KBPNew.from(julie).slotValue(canada).rel("r").KBTriple());
          add(KBPNew.from(arun).slotValue(canada).rel("r").KBTriple());
        }},
        v.asPath());
    assertEquals(
        new ArrayList<KBTriple>(){{
          add(KBPNew.from(julie).slotValue(canada).rel("r").KBTriple());
          add(KBPNew.from(arun).slotValue(canada).rel("r").KBTriple());
          add(KBPNew.from(arun).slotValue(gabor).rel("r").KBTriple());
        }},
        w.asPath());

    Props.TEST_GRAPH_INFERENCE_DEPTH = savedDepth;
  }

  @Test
  public void testAllPathsInTrie() {
    int savedDepth = Props.TEST_GRAPH_INFERENCE_DEPTH;
    Props.TEST_GRAPH_INFERENCE_DEPTH = 4;
    final Trie t = new Trie(julie);
    final Trie u = t.extend(KBPNew.from(julie).slotValue(canada).rel("r").KBTriple()).orCrash();
    final Trie v = u.extend(KBPNew.from(arun).slotValue(canada).rel("r").KBTriple()).orCrash();
    final Trie w1 = v.extend(KBPNew.from(arun).slotValue(gabor).rel("r").KBTriple()).orCrash();
    final Trie w2 = v.extend(KBPNew.from(arun).slotValue(percy).rel("r").KBTriple()).orCrash();

    assertEquals(
        new HashSet<List<KBTriple>>() {{
          add(u.asPath());
          add(v.asPath());
          add(w1.asPath());
          add(w2.asPath());
        }},
        new HashSet<>(t.allPathsInTrie())
    );

    Props.TEST_GRAPH_INFERENCE_DEPTH = savedDepth;
  }
}
