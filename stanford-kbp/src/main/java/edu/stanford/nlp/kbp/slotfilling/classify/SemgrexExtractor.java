package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.semgraph.semgrex.SemgrexBatchParser;
import edu.stanford.nlp.semgraph.semgrex.SemgrexMatcher;
import edu.stanford.nlp.semgraph.semgrex.SemgrexPattern;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.*;

/**
 * A relation extractor making use of semgrex dependency patterns
 * 
 * @author Sonal Gupta
 */
public class SemgrexExtractor extends HeuristicRelationExtractor {

  private static final long serialVersionUID = 1L;
  private final Map<RelationType, Collection<SemgrexPattern>> rules = new HashMap<>();

  public SemgrexExtractor() {
    // Create extractors
    logger.log("Reading Semgrex rules from the directory " + Props.TRAIN_SEMGREX_DIR);
    assert IOUtils.existsInClasspathOrFileSystem(Props.TRAIN_SEMGREX_DIR.toString());
    SemgrexBatchParser parser = new SemgrexBatchParser();
    for (RelationType rel : RelationType.values()) {
      String filename = Props.TRAIN_SEMGREX_DIR + File.separator + rel.canonicalName + ".rules";
      if (IOUtils.existsInClasspathOrFileSystem(filename)) {

        Counter<SemgrexPattern> rulesforrel = null;
        try {
          rulesforrel = parser.compileStream(new FileInputStream(new File(filename)));
        } catch (IOException e) {
          e.printStackTrace();
          System.exit(-1);
        }
        // for (String line : IOUtils.readLines(Props.TRAIN_SEMGREX_DIR +
        // File.separator + rel.canonicalName + ".rules")) {
        // rulesforrel.add(SemgrexPattern.compile(line));
        // }
        logger.log("Read " + rulesforrel.size() + " rules from " + filename + " for relation " + rel);
        rules.put(rel, rulesforrel.keySet());
      }
    }
  }

  public SemgrexExtractor(Properties props) {
    this();
  }

  @Override
  public Collection<Pair<String, Integer>> extractRelations(KBPair key, CoreMap[] input) {

    Map<Integer, List<Span>> entitySpans = new HashMap<Integer, List<Span>>();
    Map<Integer, List<Span>> slotSpans = new HashMap<Integer, List<Span>>();

    for (int i = 0; i < input.length; i++) {
      CoreMap sentence = input[i];
      List<Span> entitys = new ArrayList<Span>();
      List<Span> slots = new ArrayList<Span>();
      for (EntityMention entityMention : sentence.get(MachineReadingAnnotations.EntityMentionsAnnotation.class)) {
        if ((entityMention.getValue() != null && entityMention.getValue().equalsIgnoreCase(key.entityName))
            || (entityMention.getNormalizedName() != null && entityMention.getNormalizedName().equalsIgnoreCase(key.entityName))) {
          entitys.add(entityMention.getExtent());

          // for (int i = entityMention.getExtentTokenStart(); i <
          // entityMention.getExtentTokenEnd(); ++i) {
          // sentence.get(CoreAnnotations.TokensAnnotation.class).get(i).set(KBPEntity.class,
          // "true");
          // }
        }
      }
      entitySpans.put(i, entitys);

      for (EntityMention slotMention : sentence.get(KBPAnnotations.SlotMentionsAnnotation.class)) {
        //System.out.println("slot mention is " + slotMention + " and tokens are " + sentence.get(CoreAnnotations.TokensAnnotation.class));
        if ((slotMention.getValue() != null && slotMention.getValue().replaceAll("\\\\", "").equals(key.slotValue))
            || (slotMention.getNormalizedName() != null && slotMention.getNormalizedName().equalsIgnoreCase(key.slotValue))) {
          slots.add(slotMention.getExtent());
          // for (int i = slotMention.getExtentTokenStart(); i <
          // slotMention.getExtentTokenEnd(); ++i) {
          // sentence.get(CoreAnnotations.TokensAnnotation.class).get(i).set(KBPSlotFill.class,
          // "true");
          // }
        }
      }
      slotSpans.put(i, slots);
    }

    // Run Rules
    Set<Pair<String, Integer>> output = new HashSet<>();
    relationLoop: for (RelationType rel : RelationType.values()) {
//      /logger.log("Matching for relation " + rel);

      if (rules.containsKey(rel)) {
        Collection<SemgrexPattern> rulesForRel = rules.get(rel);
        for (int sentI = 0; sentI < input.length; ++sentI) {
          CoreMap sentence = input[sentI];
          boolean matches = matches(sentence, rulesForRel, key, entitySpans.get(sentI), slotSpans.get(sentI));
          if (matches) {
            //logger.log("MATCH for " + rel +  ". " + sentence: + sentence + " with rules for  " + rel);
            output.add(Pair.makePair(rel.canonicalName, sentI));
            continue relationLoop;
          }
        }
      }
    }
    return output;
  }

  boolean matches(CoreMap sentence, Collection<SemgrexPattern> rulesForRel, KBPair key, List<Span> entitySpan, List<Span> slotSpan) {
    SemanticGraph graph = sentence.get(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class);

    if (graph == null){
      logger.warn("Semantic graph is null ");
      return false;
    }
    List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
    Span matchedEntitySpan = null, matchedSlotSpan = null;
    for (SemgrexPattern p : rulesForRel) {

      try {
        //logger.log("Matching " + p + " with graph " + graph);
        SemgrexMatcher n = p.matcher(graph);
        while (n.find()) {
          IndexedWord entity = n.getNode("entity");
          IndexedWord slot = n.getNode("slot");
          //logger.log("entity is " + entity + " and slot is " + slot);
          boolean hasEntity = false, hasSlot = false;
          for (Span en : entitySpan) {
            if (entity.index() >= en.start() + 1 && entity.index() <= en.end()) {
              hasEntity = true;
              matchedEntitySpan = en;
              break;
            }
          }

          for (Span en : slotSpan) {
            if (slot.index() >= en.start() + 1 && slot.index() <= en.end()) {
              hasSlot = true;
              matchedSlotSpan = en;
              break;
            }
          }

          if (hasEntity && hasSlot) {
            for(int i = matchedEntitySpan.start(); i < matchedEntitySpan.end(); i++){
               tokens.get(i).set(KBPAnnotations.IsEntity.class, true);
            }
            for(int i = matchedSlotSpan.start(); i < matchedSlotSpan.end(); i++){
              tokens.get(i).set(KBPAnnotations.IsSlot.class, true);
            }
            return true;
          }
        }
      } catch (Exception e) {
        //Happens when graph has no roots
        return false;
      }
    }
    return false;
  }
}
