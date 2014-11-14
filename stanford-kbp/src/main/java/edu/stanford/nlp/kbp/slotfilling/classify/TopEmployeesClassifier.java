package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.common.KBPAnnotations;
import edu.stanford.nlp.kbp.common.KBPair;
import edu.stanford.nlp.kbp.common.NERTag;
import edu.stanford.nlp.kbp.common.RelationType;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

import java.util.*;

/**
 * A silly little classifier for predicting top employees based on a list of keywords in the span
 * between the two mentions.
 *
 * @author Gabor Angeli
 */
public class TopEmployeesClassifier extends HeuristicRelationExtractor {

  public static final Set<String> TOP_EMPLOYEE_TRIGGERS = Collections.unmodifiableSet(new HashSet<String>(){{
    add("executive");
    add("chairman");
    add("president");
    add("chief");
    add("head");
    add("general");
    add("ceo");
    add("officer");
    add("founder");
    add("found");
    add("leader");
    add("vice");
    add("king");
    add("prince");
    add("manager");
    add("host");
    add("minister");
    add("adviser");
    add("boss");
    add("chair");
    add("ambassador");
    add("shareholder");
    add("star");
    add("governor");
    add("investor");
    add("representative");
    add("dean");
    add("commissioner");
    add("deputy");
    add("commander");
    add("scientist");
    add("midfielder");
    add("speaker");
    add("researcher");
    add("editor");
    add("chancellor");
    add("fellow");
    add("leadership");
    add("diplomat");
    add("attorney");
    add("associate");
    add("striker");
    add("pilot");
    add("captain");
    add("banker");
    add("mayer");
    add("premier");
    add("producer");
    add("architect");
    add("designer");
    add("major");
    add("advisor");
    add("presidency");
    add("senator");
    add("specialist");
    add("faculty");
    add("monitor");
    add("chairwoman");
    add("mayor");
    add("columnist");
    add("mediator");
    add("prosecutor");
    add("entrepreneur");
    add("creator");
    add("superstar");
    add("commentator");
    add("principal");
    add("operative");
    add("businessman");
    add("peacekeeper");
    add("investigator");
    add("coordinator");
    add("knight");
    add("lawmaker");
    add("justice");
    add("publisher");
    add("playmaker");
    add("moderator");
    add("negotiator");
  }});


  public TopEmployeesClassifier() { }

  @SuppressWarnings("UnusedDeclaration")
  public TopEmployeesClassifier(@SuppressWarnings("UnusedParameters") Properties props) {
    this();
  }

  @Override
  public Collection<Pair<String, Integer>> extractRelations(KBPair key, CoreMap[] input) {
    if (!key.slotType.equalsOrElse(NERTag.PERSON, true) &&
        !(key.slotType.equalsOrElse(NERTag.ORGANIZATION, true) || key.slotType.equalsOrElse(NERTag.COUNTRY, true) ||
          key.slotType.equalsOrElse(NERTag.STATE_OR_PROVINCE, true) || key.slotType.equalsOrElse(NERTag.CITY, true))) {
      //noinspection unchecked
      return Collections.EMPTY_SET;
    }
    Collection<Pair<String,Integer>> extractions = new HashSet<>();

    for (int sentI = 0; sentI < input.length; ++sentI) {
      // Get where the entity is
      Span entitySpan = null;
      for (EntityMention entityMention : input[sentI].get(MachineReadingAnnotations.EntityMentionsAnnotation.class)) {
        if ((entityMention.getValue() != null && entityMention.getValue().equalsIgnoreCase(key.entityName)) ||
            (entityMention.getNormalizedName() != null && entityMention.getNormalizedName().equalsIgnoreCase(key.entityName))) {
          entitySpan = new Span(entityMention.getExtentTokenStart(), entityMention.getExtentTokenEnd());
        }
      }
      // Get where the slot fill is
      Span slotValueSpan = null;
      for (EntityMention slotMention : input[sentI].get(KBPAnnotations.SlotMentionsAnnotation.class)) {
        if ((slotMention.getValue() != null && slotMention.getValue().replaceAll("\\\\", "").equals(key.slotValue)) ||
            (slotMention.getNormalizedName() != null && slotMention.getNormalizedName().equalsIgnoreCase(key.slotValue))) {
          slotValueSpan = new Span(slotMention.getExtentTokenStart(), slotMention.getExtentTokenEnd());
        }
      }

      if (entitySpan != null && slotValueSpan != null && !Span.overlaps(entitySpan, slotValueSpan)) {
        int betweenStart = Math.min(entitySpan.end(), slotValueSpan.end());
        int betweenEnd   = Math.max(entitySpan.start(), slotValueSpan.start());
        assert betweenStart <= betweenEnd;
        if (betweenEnd - betweenStart < 5) {
          List<CoreLabel> tokens = input[sentI].get(CoreAnnotations.TokensAnnotation.class);

          // Check if span is broken by punctuation
          boolean brokenByPunctuation = false;
          boolean brokenByPERorORG = false;
          for (int i = betweenStart; i < betweenEnd; ++i) {
            if (tokens.get(i).originalText().equals(",") || tokens.get(i).originalText().equals(";") ||
                tokens.get(i).originalText().equals("\"")) {
              brokenByPunctuation = true;
            }
            if (tokens.get(i).ner().equals(NERTag.PERSON.name) || tokens.get(i).ner().equals(NERTag.ORGANIZATION.name) ||
                tokens.get(i).ner().equals(NERTag.COUNTRY.name) || tokens.get(i).ner().equals(NERTag.STATE_OR_PROVINCE.name) ||
                tokens.get(i).ner().equals(NERTag.CITY.name)) {
              brokenByPERorORG = true;
            }
          }

          // Don't consider if there's a PER or ORG in the way
          if (brokenByPERorORG) {
            continue;
          }

          // Look for keywords
          for (int i = betweenStart; i < betweenEnd; ++i) {
            if (!tokens.get(i).tag().startsWith("V") &&
                (TOP_EMPLOYEE_TRIGGERS.contains(tokens.get(i).originalText().toLowerCase()) ||
                 TOP_EMPLOYEE_TRIGGERS.contains(tokens.get(i).lemma().toLowerCase()))) {
              if (key.entityType == NERTag.PERSON &&
                  (key.slotType.equalsOrElse(NERTag.ORGANIZATION, true) ||
                   key.slotType.equalsOrElse(NERTag.COUNTRY, true) || key.slotType.equalsOrElse(NERTag.STATE_OR_PROVINCE, true) || key.slotType.equalsOrElse(NERTag.CITY, true))) {
                extractions.add(Pair.makePair(RelationType.PER_EMPLOYEE_OF.canonicalName, sentI));
              } else if (key.entityType == NERTag.ORGANIZATION && key.slotType.equalsOrElse(NERTag.PERSON, true)) {
                if (slotValueSpan.end() < entitySpan.start() || !brokenByPunctuation) {
                  // ^^ prohibit things like "something something Organization, said president Obama"
                  extractions.add(Pair.makePair(RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES.canonicalName, sentI));
                }
              }
            }
          }
        }
      }

    }

    return extractions;
  }
}
