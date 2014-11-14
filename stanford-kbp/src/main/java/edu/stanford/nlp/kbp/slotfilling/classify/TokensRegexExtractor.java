package edu.stanford.nlp.kbp.slotfilling.classify;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations;
import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.tokensregex.CoreMapExpressionExtractor;
import edu.stanford.nlp.ling.tokensregex.TokenSequencePattern;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.File;
import java.io.FilenameFilter;
import java.util.*;
import java.util.stream.Collectors;

/**
 * A relation extractor making use of simple token regex patterns
 *
 * @author Gabor Angeli
 */
public class TokensRegexExtractor extends HeuristicRelationExtractor {
  protected final Redwood.RedwoodChannels logger = Redwood.channels("TokensRegex");
  private final Map<RelationType, CoreMapExpressionExtractor> rules = new HashMap<>();

  public TokensRegexExtractor() {
    logger.log("Creating TokensRegexExtractor");
    // Create extractors
    for (RelationType rel : RelationType.values()) {

      FilenameFilter filter = (dir, name) -> {
        if(name.matches(rel.canonicalName.replaceAll("/", "SLASH") + ".*.rules"))
          return true;
        return false;
      };

      if (Props.TRAIN_TOKENSREGEX_DIR.exists()){
        File[] ruleFiles = Props.TRAIN_TOKENSREGEX_DIR.listFiles(filter);

        if(ruleFiles != null && ruleFiles.length > 0){
          List<String> listfiles = new ArrayList<String>();
          listfiles.add(Props.TRAIN_TOKENSREGEX_DIR + File.separator + "defs.rules");
          listfiles.addAll(Arrays.asList(ruleFiles).stream().map( f -> f.getAbsolutePath()).collect(Collectors.toList()));
          logger.log("Rule files for relation " + rel + " are " + listfiles);
          rules.put(rel, CoreMapExpressionExtractor.createExtractorFromFiles(TokenSequencePattern.getNewEnv(), listfiles));
        }
      }
//      if (IOUtils.existsInClasspathOrFileSystem(Props.TRAIN_TOKENSREGEX_DIR + File.separator + rel.canonicalName + ".rules")) {
//        rules.put(rel, CoreMapExpressionExtractor.createExtractorFromFiles(TokenSequencePattern.getNewEnv(),
//            Props.TRAIN_TOKENSREGEX_DIR + File.separator + "defs.rules",
//            Props.TRAIN_TOKENSREGEX_DIR + File.separator + rel.canonicalName + ".rules"));
//      }
    }
  }

  public TokensRegexExtractor(@SuppressWarnings("UnusedParameters") Properties props) {
    this();
  }

  @Override
  public Collection<Pair<String,Integer>> extractRelations(KBPair key, CoreMap[] input) {
    startTrack("Extracting using TokensRegex");
    /// Sanity Check
    if (Utils.assertionsEnabled()) {
      for (CoreMap sentence : input) {
        for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
          assert !token.containsKey(KBPEntity.class);
          assert !token.containsKey(KBPSlotFill.class);
        }
      }
    }

    // Annotate Sentence
    for (CoreMap sentence : input) {
      // Annotate where the entity is
      for (EntityMention entityMention : sentence.get(MachineReadingAnnotations.EntityMentionsAnnotation.class)) {
        if ((entityMention.getValue() != null && entityMention.getValue().equalsIgnoreCase(key.entityName)) ||
            (entityMention.getNormalizedName() != null && entityMention.getNormalizedName().equalsIgnoreCase(key.entityName))) {
          for (int i = entityMention.getExtentTokenStart(); i < entityMention.getExtentTokenEnd(); ++i) {
            sentence.get(CoreAnnotations.TokensAnnotation.class).get(i).set(KBPEntity.class, "true");
          }
        }
      }
      // Annotate where the slot fill is
      for (EntityMention slotMention : sentence.get(KBPAnnotations.SlotMentionsAnnotation.class)) {
        if ((slotMention.getValue() != null && slotMention.getValue().replaceAll("\\\\", "").equals(key.slotValue)) ||
            (slotMention.getNormalizedName() != null && slotMention.getNormalizedName().equalsIgnoreCase(key.slotValue))) {
          for (int i = slotMention.getExtentTokenStart(); i < slotMention.getExtentTokenEnd(); ++i) {
            sentence.get(CoreAnnotations.TokensAnnotation.class).get(i).set(KBPSlotFill.class, "true");
          }
        }
      }
    }
    // Run Rules
    Set<Pair<String,Integer>> output = new HashSet<>();
    relationLoop: for (RelationType rel : RelationType.values()) {
      if (rules.containsKey(rel)) {
        CoreMapExpressionExtractor extractor = rules.get(rel);
        for (int sentI = 0; sentI < input.length; ++sentI) {
          CoreMap sentence = input[sentI];

          List extractions = extractor.extractExpressions(sentence);
          if (extractions != null && extractions.size() > 0) {
            logger.log("matched " + sentence + " with rules for " + rel);
            output.add(Pair.makePair(rel.canonicalName, sentI));
            continue relationLoop;
          }
        }
      }
    }

    // Un-Annotate Sentence
    for (CoreMap sentence : input) {
      for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
        token.remove(KBPEntity.class);
        token.remove(KBPSlotFill.class);
      }
    }
    endTrack("Extracting using TokensRegex");
    return output;
  }

  public static class KBPEntity implements CoreAnnotation<String> {
    public Class<String> getType() { return String.class; }
  }

  public static class KBPSlotFill implements CoreAnnotation<String> {
    public Class<String> getType() { return String.class; }
  }
}
