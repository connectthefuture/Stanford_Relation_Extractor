package edu.stanford.nlp.kbp.slotfilling.spec;

import edu.stanford.nlp.kbp.common.Maybe;
import edu.stanford.nlp.kbp.common.RelationType;

import java.io.IOException;
import java.io.StringReader;
import java.util.HashSet;
import java.util.List;

import org.junit.Test;
import org.xml.sax.SAXException;

import edu.stanford.nlp.kbp.common.NERTag;
import edu.stanford.nlp.kbp.common.KBPOfficialEntity;

import static junit.framework.Assert.*;

public class TaskXMLParserTest {
  /**
   * Checks that the EntityMention passed in has the expected values
   */
  public void checkMention(KBPOfficialEntity mention,
                           String queryId, String name,
                           NERTag type, String nodeId,
                           String ... ignorables) {
    assertEquals(type, mention.type);
    assertEquals(name, mention.name);
    assertEquals(nodeId, mention.id.orCrash());
    assertEquals(queryId, mention.queryId.orCrash());
    assertEquals(ignorables.length, mention.ignoredSlots.getOrElse(new HashSet<RelationType>()).size());
    for (String ignoredSlot : ignorables) {
      assertTrue(mention.ignoredSlots.getOrElse(new HashSet<RelationType>()).contains(RelationType.fromString(ignoredSlot).orCrash()));
    }
  }
                           
  /**
   * Checks that a document which should be parsed correctly is parsed correctly
   */
  @Test
  public void testParse() 
    throws Exception
  {
    List<KBPOfficialEntity> mentions =
      TaskXMLParser.parseQueryFile(new StringReader(testDoc), Maybe.<edu.stanford.nlp.kbp.slotfilling.ir.KBPIR>Nothing());
    assertEquals(3, mentions.size());
    checkMention(mentions.get(0), "SF213", "Viacom", 
                 NERTag.ORGANIZATION, "NIL00014",
                 "org:founded", "org:country_of_headquarters",
                 "org:city_of_headquarters", "org:website");
    checkMention(mentions.get(1), "SF214", "Paul Newman",
                 NERTag.PERSON, "E0181364");
    checkMention(mentions.get(2), "SF215", "Francois Mitterrand",
                 NERTag.PERSON, "NIL00015");
  }

  /**
   * Checks that we throw a SAXException on the given doc
   */
  public void checkThrowsSAXException(String doc)
    throws IOException
  {
    try {
      TaskXMLParser.parseQueryFile(new StringReader(doc), Maybe.<edu.stanford.nlp.kbp.slotfilling.ir.KBPIR>Nothing());
    } catch (SAXException e) {
      return;
    }
    throw new RuntimeException("Expected a SAXException");
  }

  /**
   * Checks that various error cases are caught
   */
  @Test
  public void testExceptions() 
    throws IOException
  {
    checkThrowsSAXException(noQueryIdDoc);
    checkThrowsSAXException(noNameDoc);
    checkThrowsSAXException(noNodeIdDoc);
  }

  public void testClean()
    throws IOException, SAXException
  {
    // should not actually barf now that we skip these
    // TODO: test for empty doc
    TaskXMLParser.parseQueryFile(new StringReader(noEntTypeDoc), Maybe.<edu.stanford.nlp.kbp.slotfilling.ir.KBPIR>Nothing());
  }

  /**
   * This check sees what happens when the &lt;ignore&gt; field is an
   * empty string
   */
  @Test
  public void testEmptyIgnore() 
    throws IOException, SAXException
  {
    List<KBPOfficialEntity> mentions =
      TaskXMLParser.parseQueryFile(new StringReader(emptyIgnoreDoc), Maybe.<edu.stanford.nlp.kbp.slotfilling.ir.KBPIR>Nothing());
    checkMention(mentions.get(0), "SF213", "Viacom", 
                 NERTag.ORGANIZATION, "NIL00014");
  }

  static public final String noQueryIdDoc = "<?xml version='1.0' encoding='UTF-8'?>\n<kbpslotfill>\n  <query>\n    <name>Viacom</name>\n    <docid>eng-WL-11-174596-12958306</docid>\n    <enttype>ORG</enttype>\n    <nodeid>NIL00014</nodeid>\n    <ignore>org:founded org:country_of_headquarters org:city_of_headquarters org:website</ignore>\n  </query>\n</kbpslotfill>";

  static public final String noNameDoc = "<?xml version='1.0' encoding='UTF-8'?>\n<kbpslotfill>\n  <query id=\"foo\">\n    <docid>eng-WL-11-174596-12958306</docid>\n    <enttype>ORG</enttype>\n    <nodeid>NIL00014</nodeid>\n    <ignore>org:founded org:country_of_headquarters org:city_of_headquarters org:website</ignore>\n  </query>\n</kbpslotfill>";

  static public final String noNodeIdDoc = "<?xml version='1.0' encoding='UTF-8'?>\n<kbpslotfill>\n  <query id=\"foo\">\n    <name>Viacom</name>\n    <docid>eng-WL-11-174596-12958306</docid>\n    <enttype>ORG</enttype>\n    <ignore>org:founded org:country_of_headquarters org:city_of_headquarters org:website</ignore>\n  </query>\n</kbpslotfill>";

  static public final String noEntTypeDoc = "<?xml version='1.0' encoding='UTF-8'?>\n<kbpslotfill>\n  <query id=\"foo\">\n    <name>Viacom</name>\n    <docid>eng-WL-11-174596-12958306</docid>\n    <nodeid>NIL00014</nodeid>\n    <ignore>org:founded org:country_of_headquarters org:city_of_headquarters org:website</ignore>\n  </query>\n</kbpslotfill>";

  static public final String testDoc = "<?xml version='1.0' encoding='UTF-8'?>\n<kbpslotfill>\n  <query id=\"SF213\">\n    <name>Viacom</name>\n    <docid>eng-WL-11-174596-12958306</docid>\n    <enttype>ORG</enttype>\n    <nodeid>NIL00014</nodeid>\n    <ignore>org:founded org:country_of_headquarters org:city_of_headquarters org:website</ignore>\n  </query>\n  <query id=\"SF214\">\n    <name>Paul Newman</name>\n    <docid>eng-WL-11-174596-12959584</docid>\n    <enttype>PER</enttype>\n    <nodeid>E0181364</nodeid>\n  </query>\n  <query id=\"SF215\">\n    <name>Francois Mitterrand</name>\n    <docid>eng-WL-11-174596-12959584</docid>\n    <enttype>PER</enttype>\n    <nodeid>NIL00015</nodeid>\n  </query>\n</kbpslotfill>";

  static public final String emptyIgnoreDoc = "<?xml version='1.0' encoding='UTF-8'?>\n<kbpslotfill>\n  <query id=\"SF213\">\n    <name>Viacom</name>\n    <docid>eng-WL-11-174596-12958306</docid>\n    <enttype>ORG</enttype>\n    <nodeid>NIL00014</nodeid>\n    <ignore/>\n  </query>\n</kbpslotfill>";
}