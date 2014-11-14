package edu.stanford.nlp.kbp.slotfilling.spec;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.PostIRAnnotator;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.MetaClass;
import edu.stanford.nlp.util.TagStackXmlHandler;
import org.xml.sax.Attributes;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

public class TaskXMLParser extends TagStackXmlHandler {
  static final String[] NEW_QUERY_TAGS = {"kbpslotfill", "query"};
  static final String[] NAME_TAGS = {"kbpslotfill", "query", "name"};
  static final String[] DOCID_TAGS = {"kbpslotfill", "query", "docid"};
  // treebeard, leaflock, beechbone, etc
  static final String[] ENTTYPE_TAGS = {"kbpslotfill", "query", "enttype"};
  static final String[] NODEID_TAGS = {"kbpslotfill", "query", "nodeid"};
  static final String[] IGNORE_TAGS = {"kbpslotfill", "query", "ignore"};

  static final String ID_ATTRIBUTE = "id";
  
  //sometimes the query file has node id as nil. Assign them nil+nilIDCounter;
  static int nilIDCounter = 0;

  /**
   * Returns a list of the EntityMentions contained in the Reader passed in.
   * <br>
   * This can throw exceptions in the following circumstances:
   * <br>
   * If there is a nested &lt;query&gt; tag, it will throw a SAXException
   * <br>
   * If there is a &lt;query&gt; tag with no id attribute, it will also throw
   * a SAXException
   * <br>
   * If any of the name, enttype, or nodeid fields are missing, it once
   * again throws a SAXException
   * <br>
   * If there is a problem with the reader passed in, it may throw an
   * IOException
   */
  public static List<KBPOfficialEntity> parseQueryFile(Reader input, Maybe<KBPIR> ir)
    throws IOException, SAXException
  {
    InputSource source = new InputSource(input);
    source.setEncoding("UTF-8");
    
    TaskXMLParser handler = new TaskXMLParser(ir);
    
    try {
      SAXParser parser = SAXParserFactory.newInstance().newSAXParser();
      parser.parse(source, handler);
    } catch(ParserConfigurationException e) {
      throw new RuntimeException(e);
    }
    return handler.mentions;
  }

  public static List<KBPOfficialEntity> parseQueryFile(String filename, Maybe<KBPIR> ir)
    throws IOException, SAXException
  {
    BufferedReader reader = IOUtils.getBufferedReaderFromClasspathOrFileSystem(filename);
    List<KBPOfficialEntity> mentions = parseQueryFile(reader, ir);
    reader.close();
    return mentions;
  }

  private final Maybe<KBPIR> ir;

  /**
   * The only way to use one of these objects is through the
   * parseQueryFile method
   */
  private TaskXMLParser(Maybe<KBPIR> ir ) { this.ir = ir; }

  List<KBPOfficialEntity> mentions = new ArrayList<>();

//  KBPOfficialEntity currentMention = null;
  Map<String, String> currentMention = new HashMap<>();
  StringBuilder currentText = null;

  @Override
  public void startElement(String uri, String localName, 
                           String qName, Attributes attributes)
    throws SAXException
  {
    super.startElement(uri, localName, qName, attributes);

    if (matchesTags(NEW_QUERY_TAGS)) {
      if (!currentMention.isEmpty())
        throw new RuntimeException("Unexpected nested query after query #" + 
                                   mentions.size());
      currentMention = new HashMap<>();
      String id = attributes.getValue(ID_ATTRIBUTE);
      debug("Query ID is " + id);
      if (id == null) 
        throw new SAXException("Query #" + (mentions.size() + 1) + 
                               " has no id, " +
                               "what are we supposed to do with that?");
      currentMention.put("queryId", id);
    } else if (matchesTags(NAME_TAGS) || matchesTags(DOCID_TAGS) ||
               matchesTags(ENTTYPE_TAGS) || matchesTags(NODEID_TAGS) ||
               matchesTags(IGNORE_TAGS)) {
      currentText = new StringBuilder();
    }
  }
  
  @Override
  public void endElement(String uri, String localName, 
                         String qName) 
    throws SAXException
  {
    if (currentText != null) {
      String text = currentText.toString().trim();
      if (matchesTags(NAME_TAGS)) {
        currentMention.put("name", text);
      } else if (matchesTags(DOCID_TAGS)) {
        currentMention.put("docid", text);
      } else if (matchesTags(ENTTYPE_TAGS)) {
        currentMention.put("type", text);
      } else if (matchesTags(NODEID_TAGS)) {
        currentMention.put("id", text);
      } else if (matchesTags(IGNORE_TAGS)) {
        if (!text.equals("")) {
          currentMention.put("ignoredSlots", text);
//          String[] ignorables = text.split("\\s+");
//          Set<String> ignoredSlots = new HashSet<String>();
//          for (String ignore : ignorables) {
//            ignoredSlots.add(ignore);
//          }
        }
      } else {
        throw new RuntimeException("Programmer error!  " + 
                                   "Tags handled in startElement are not " +
                                   "handled in endElement");
      }
      currentText = null;
    }
    if (matchesTags(NEW_QUERY_TAGS)) {
      boolean shouldAdd = true;
      if (currentMention == null) {
        throw new NullPointerException("Somehow exited a query block with " +
                                       "currentMention set to null");
      }
      if (!currentMention.containsKey("ignoredSlots")) {
        currentMention.put("ignoredSlots", "");
      }
      if (!currentMention.containsKey("type")) {
        System.err.println("Query #" + (mentions.size() + 1) +
                           " has no known type. It was probably GPE. Skipping...");
        shouldAdd = false;
      } 
      if (!currentMention.containsKey("name")) {
        throw new SAXException("Query #" + (mentions.size() + 1) +
                               " has no name");
      } 
      if ((Props.KBP_YEAR == Props.YEAR.KBP2009 ||
          Props.KBP_YEAR == Props.YEAR.KBP2010 ||
          Props.KBP_YEAR == Props.YEAR.KBP2011 ||
          Props.KBP_YEAR == Props.YEAR.KBP2012 ||
          Props.KBP_YEAR == Props.YEAR.KBP2013) &&
          !currentMention.containsKey("id")) {
        throw new SAXException("Query #" + (mentions.size() + 1) +
                               " has no nodeid");
      } 
      if (!currentMention.containsKey("queryId")) {
        throw new SAXException("Query #" + (mentions.size() + 1) +
                               " has no queryid");
      }
      if(!currentMention.containsKey("id") || currentMention.get("id").equals("NIL"))
      {
        String newId = "NIL"+nilIDCounter;
        warn("query " + currentMention.get("queryId") + " has id as NIL. Assigning it random id " + nilIDCounter + " (OK if this is the official evaluation!)");
        currentMention.put("id", newId);
        nilIDCounter ++;
      }
      if(shouldAdd) {
        String[] ignoredSlots = MetaClass.cast(currentMention.get("ignoredSlots"), String[].class);
        Set<RelationType> ignoredRelationSlots = new HashSet<>();
        for (String rel : ignoredSlots) { ignoredRelationSlots.add(RelationType.fromString(rel).orCrash()); }

        Maybe<EntityContext> representativeDocument = Maybe.Nothing();
        for (KBPIR querier : ir) {
          KBPEntity simpleEntity = KBPNew.entName(currentMention.get("name")).entType(currentMention.get("type")).KBPEntity();
          Annotation doc = null;
          Maybe<Span> tokenSpan = Maybe.Nothing();
          Maybe<Integer> sentenceIndex = Maybe.Nothing();

          if (Props.INDEX_MODE != Props.QueryMode.NOOP) {

            // Get the representative document
            doc = querier.fetchDocument(currentMention.get("docid"), true);
            final PostIRAnnotator postIRAnnotator = new PostIRAnnotator(KBPNew.from(simpleEntity).KBPOfficialEntity(), true);
            postIRAnnotator.annotate(doc);

            // Get the representative span inside it
            List<CoreMap> sentences = doc.get(CoreAnnotations.SentencesAnnotation.class);
            // -- pass 1: exact match
            for (int s = 0; s < sentences.size(); ++s) {
              CoreMap sentence = sentences.get(s);
              if (!tokenSpan.isDefined()) {
                tokenSpan = Utils.getTokenSpan(sentence.get(CoreAnnotations.TokensAnnotation.class), simpleEntity.name, Maybe.<Span>Nothing());
                sentenceIndex = Maybe.Just(s);
              }
            }
            // -- pass 2: antecedent match
            for (int s = 0; s < sentences.size(); ++s) {
              CoreMap sentence = sentences.get(s);
              if (sentence.get(KBPAnnotations.AllAntecedentsAnnotation.class).contains(simpleEntity.name)) {
                List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
                int begin = -1;
                int end = -1;
                for (int i = 0; i < tokens.size(); ++i) {
                  CoreLabel token = tokens.get(i);
                  if (begin < 0 && simpleEntity.name.equals(token.get(CoreAnnotations.AntecedentAnnotation.class))) {
                    begin = i;
                  } else if (begin >= 0 && end < 0 && !simpleEntity.name.equals(token.get(CoreAnnotations.AntecedentAnnotation.class))) {
                    end = i;
                  }
                }
                if (end < 0) {
                  end = tokens.size();
                }
                if (begin >= 0) {
                  if (!tokenSpan.isDefined()) {
                    tokenSpan = Maybe.Just(new Span(begin, end));
                    sentenceIndex = Maybe.Just(s);
                  }
                }
              }
            }
          }

          // -- Check for errors
          if (!tokenSpan.isDefined() || !sentenceIndex.isDefined()) {
            err(RED, "Could not find representative mention in source document (is the linker too strict?)");
            representativeDocument = Maybe.Just(new EntityContext(simpleEntity));
          } else {
            representativeDocument = Maybe.Just(new EntityContext(simpleEntity, doc, sentenceIndex.get(), tokenSpan.get()));
          }
        }


        mentions.add(KBPNew.entName(currentMention.get("name"))
                           .entType(currentMention.get("type"))
                           .entId(currentMention.get("id"))
                           .queryId(currentMention.get("queryId"))
                           .ignoredSlots(ignoredRelationSlots)
                           .representativeDocument(representativeDocument).KBPOfficialEntity());
      }
      currentMention = new HashMap<>();
    }

    super.endElement(uri, localName, qName);
  }

  /**
   * If we're in a set of tags where we care about the text, save
   * the text.  If we're in a set of tags where we remove the
   * underscores, do that first.
   */
  @Override
  public void characters(char buf[], int offset, int len) {
    if (currentText != null) {
      currentText.append(new String(buf, offset, len));
    }
  }
  
}
