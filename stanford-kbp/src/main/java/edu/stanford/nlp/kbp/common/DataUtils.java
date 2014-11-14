package edu.stanford.nlp.kbp.common;

import edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.spec.TaskXMLParser;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.util.Quadruple;
import edu.stanford.nlp.util.Triple;
import org.xml.sax.SAXException;

import java.io.*;
import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * A utility class for reading and writing data.
 *
 * @author Gabor Angeli
 */
@SuppressWarnings("UnusedDeclaration")
public class DataUtils {
  /**
   * <p>Fetch a list of files in a directory. If the directory does not exist, optionally create it.
   * If the directory is a single file, return a singleton list with that file.</p>
   *
   * <p>Hidden files and temporary swap files are attempted to be ignored.</p>
   *
   * @param path The directory path
   * @param extension The extension of the files to retrieve
   * @param create If true, create the directory.
   * @return The list of files in that directory with the given extension, sorted by absolute path.
   */
  public static List<File> fetchFiles(String path, final String extension, boolean create) {
    File kbDir = new File(path);
    if (!kbDir.exists()) {
      if (!create) { return Collections.emptyList(); }
      if(!kbDir.mkdirs()) {
        try {
          fatal("unable to make directory " + kbDir.getCanonicalPath() + "!");
        } catch(IOException e) { fatal(e);}
      }
    }
    if(!kbDir.isDirectory()) {
      if (kbDir.getName().endsWith(extension)) { return Collections.singletonList(kbDir); }
    }
    File[] inputFiles = kbDir.listFiles(pathname -> {
      String absolutePath = pathname.getAbsolutePath();
      String filename = pathname.getName();
      return absolutePath.endsWith(extension) && !filename.startsWith(".") && !filename.endsWith("~");
    });
    List<File> files = Arrays.asList(inputFiles);
    Collections.sort(files, (o1, o2) -> o1.getAbsolutePath().compareTo(o2.getAbsolutePath()));
    return files;
  }


  /**
   * Save a {@link Properties} object to a path.
   * @param props The properties object to save.
   * @param location The location to save the properties to.
   * @throws IOException If the file is not writable
   */
  public static void saveProperties(Properties props, File location) throws IOException {
    PrintStream os = new PrintStream(new FileOutputStream(location.getAbsolutePath()));
    List<String> keys = new ArrayList<>(props.stringPropertyNames());
    Collections.sort(keys);
    for (Object key : keys) {
      os.println(key.toString() + " = " + props.get(key).toString());
    }
    os.close();
  }

  /**
   * Get the test entities.
   * This in part parses the XML file, and then makes sure that the entity is something
   * that we would extract from at least the target document given.
   * @param queryFile The query XML file
   * @param querierMaybe An optional KBPIR. Without this, the entity canonicalization will not happen.
   * @return A list of KBP Entities
   */
  public static List<KBPOfficialEntity> testEntities(String queryFile, Maybe<KBPIR> querierMaybe) {
    forceTrack("Parsing Test XML File");
    try {
      return TaskXMLParser.parseQueryFile(queryFile, querierMaybe);
    } catch (IOException | SAXException e) {
      throw new RuntimeException(e);
    } finally {
      endTrack("Parsing Test XML File");
    }
  }

  /**
   * <p>Key:
   * </p>
   * <ul>
   *   <li>Entity (e.g., Obama)</li>
   *   <li>Relation (e.g., born_in)</li>
   *   <li>Slot value (e.g., Hawaii)</li>
   *   <li>Equivalence class</li>
   * </ul>
   * <p>Value:
   * </p>
   * <ul>
   *   <li>Set of correct provenances</li>
   * </ul>
   */
  public static class GoldResponses extends HashMap<Quadruple<KBPOfficialEntity, RelationType, String, Integer>, Set<String>> {

    public Set<String> goldProvenances(KBTriple key) {
      Triple<KBPOfficialEntity, RelationType, String> x = Triple.makeTriple(KBPNew.from(key.getEntity()).KBPOfficialEntity(), key.kbpRelation(), key.slotValue);
      if (containsKey(x)) { return get(x); }
      else { return new HashSet<>(); }
    }
    public Set<String> goldProvenances(KBPSlotFill fill) { return goldProvenances(fill.key); }
    public boolean isCorrect(KBTriple key) { return !goldProvenances(key).isEmpty(); }
    public boolean isCorrect(KBPSlotFill fill) { return !goldProvenances(fill).isEmpty(); }
    public Set<KBTriple> correctFills() {
      Set<KBTriple> correct = new HashSet<>();
      for (Map.Entry<Quadruple<KBPOfficialEntity, RelationType, String, Integer>, Set<String>> entry : entrySet()) {
        if (!entry.getValue().isEmpty()) {
          correct.add(KBPNew.from(entry.getKey().first).slotValue(entry.getKey().third).rel(entry.getKey().second).KBTriple());
        }
      }
      return correct;
    }
    public Set<KBTriple> incorrectFills() {
      Set<KBTriple> incorrect = new HashSet<>();
      for (Map.Entry<Quadruple<KBPOfficialEntity, RelationType, String, Integer>, Set<String>> entry : entrySet()) {
        if (entry.getValue().isEmpty()) {
          incorrect.add(KBPNew.from(entry.getKey().first).slotValue(entry.getKey().third).rel(entry.getKey().second).KBTriple());
        }
      }
      return incorrect;
    }
  }

  public static Map<KBPEntity, Counter<String>> goldProvenancesByEntity(GoldResponseSet provenances) {
    Map<KBPEntity, Counter<String>> provenancesByEntity = new LinkedHashMap<>();
    for (Map.Entry<KBPSlotFill, Set<String>> query: provenances.correctProvenances().entrySet()) {
      KBPEntity entity = query.getKey().key.getEntity();
      Set<String> queryDocs = query.getValue();
      Counter<String> docCounts = provenancesByEntity.get(entity);
      if (docCounts == null) { provenancesByEntity.put(entity, docCounts = new IntCounter<>()); }
      for (String doc: queryDocs) {
        docCounts.incrementCount(doc);
      }
    }
    return provenancesByEntity;
  }
}
