package edu.stanford.nlp.kbp.slotfilling.ir.query;

import edu.stanford.nlp.kbp.common.NERTag;
import edu.stanford.nlp.kbp.common.Maybe;
import edu.stanford.nlp.kbp.common.Props;
import edu.stanford.nlp.kbp.common.RelationType;
import edu.stanford.nlp.util.CollectionValuedMap;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Lists of keywords for use with querying lucene
 *
 * @author Angel Chang
 */
public class Keywords {
  protected static final Redwood.RedwoodChannels logger = Redwood.channels("IR Keywords");

  private Keywords() {
    try {
      if (Props.INDEX_RELATIONTRIGGERS != null && Props.INDEX_RELATIONTRIGGERS.exists() && Props.INDEX_RELATIONTRIGGERS.canRead()) {
        loadRelationKeywords(Props.INDEX_RELATIONTRIGGERS);
        isDefined = true;
      } else {
        isDefined = false;
      }
    } catch (IOException ex) {
      throw new RuntimeException("Cannot read relation keywords from " + relationKeywords, ex);
    }

  }

  /**
   * Keywords for relations
   */
  public final boolean isDefined;
  private final CollectionValuedMap<RelationType, String> relationKeywords = new CollectionValuedMap<RelationType,String>();
  private final CollectionValuedMap<NERTag, String> relationKeywordsForNERTag = new CollectionValuedMap<NERTag,String>();

  public Collection<String> getRelationKeywords(RelationType relationType) {
    return relationKeywords.get(relationType);
  }

  public Collection<String> getRelationKeywords(NERTag entityType) {
    return relationKeywordsForNERTag.get(entityType);
  }

  private void loadRelationKeywords(File filename) throws IOException {
    BufferedReader is = new BufferedReader(new FileReader(filename));
    String line;
    while ((line = is.readLine()) != null) {
      line = line.trim();
      int firstTab = line.indexOf('\t');
      if (firstTab < 0) {
        firstTab = line.indexOf(' ');
      }
      assert (firstTab > 0 && firstTab < line.length());
      String name = line.substring(0, firstTab).trim();
      String keyword = line.substring(firstTab).trim();
      RelationType relationType = RelationType.fromString(name).getOrElse(null);
      if (relationType != null) {
        if (keyword.length() > 0)  {
          relationKeywords.add(relationType, keyword);
          relationKeywordsForNERTag.add(relationType.entityType, keyword);
        }
      } else {
        logger.warn("Unknown relation " + name);
      }
    }
    is.close();
  }


  private static Maybe<Keywords> staticInstance = Maybe.Nothing();
  public static Keywords get() {
    if (!staticInstance.isDefined()) {
      staticInstance = Maybe.Just(new Keywords());
    }
    return staticInstance.get();

  }

}
