package edu.stanford.nlp.kbp.slotfilling.ir.query;

import com.esotericsoftware.kryo.KryoException;
import edu.stanford.nlp.kbp.common.Maybe;
import edu.stanford.nlp.kbp.common.Props;
import edu.stanford.nlp.kbp.slotfilling.ir.index.KBPField;
import edu.stanford.nlp.kbp.slotfilling.ir.index.KryoAnnotationSerializer;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationSerializer;
import edu.stanford.nlp.util.Pair;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.util.BytesRef;

import java.io.*;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Simple interface to get fields from a Lucene document
 *   depending on version of KBP index format used
 *
 * @author Angel Chang
 */
public interface LuceneDocumentReader {
  public String getDocidField();
  public String getDocid(Document doc);
  public Maybe<Annotation> getAnnotation(Document doc) throws ClassNotFoundException, IOException;

  public enum KBPIndexVersion {
    KBP_INDEX_2013(new Kbp2013LuceneReader());

    public LuceneDocumentReader reader;
    KBPIndexVersion(LuceneDocumentReader reader) { this.reader = reader; }

    public static KBPIndexVersion fromReader(IndexReader reader) {
      try {
        KBPIndexVersion found = null;
        for (KBPIndexVersion version : values()) {
          if (LuceneUtils.hasField(reader, version.reader.getDocidField())) {
            // We found an index that's consistent with this version
            if (found != null) {
              LuceneQuerier.logger.warn("Could not disambiguate index version: " + reader);
              return Props.INDEX_DEFAULTVERSION;
            } else {
              LuceneQuerier.logger.debug("using lucene version: " + version + " for " + reader);
              found = version;
            }
          }
        }
        if (found == null) {
          LuceneQuerier.logger.warn("Could not determine index type; using default");
          return Props.INDEX_DEFAULTVERSION;
        } else {
          return found;
        }
      } catch (IOException e) {
        LuceneQuerier.logger.warn(e);
        return Props.INDEX_DEFAULTVERSION;
      }
    }
  }

  // New KBP index 2013 Reader
  public static class Kbp2013LuceneReader implements LuceneDocumentReader {
    // Serializer has to match what ever was used to serialize the annotations
    Map<Long, AnnotationSerializer> serializerMap = new ConcurrentHashMap<>();

    public String getDocidField() {
      return KBPField.DOCID.fieldName();
    }

    public String getDocid(Document doc) {
      return doc.get(getDocidField());
    }

    public Maybe<Annotation> getAnnotation(Document doc) throws ClassNotFoundException, IOException  {
      if (serializerMap.size() > 100) {
        serializerMap.clear();
      }
      AnnotationSerializer serializer = serializerMap.get(Thread.currentThread().getId());
      if (serializer == null) {
        LuceneQuerier.logger.log("creating Kryo serializer on thread " + Thread.currentThread().getId());
        serializer = new KryoAnnotationSerializer(true, false, !Props.HACKS_OLDINDEXSERIALIZATION);
        serializerMap.put(Thread.currentThread().getId(), serializer);
      }
      if (doc == null) { return Maybe.Nothing(); }
      String coreMapVersion = doc.get(KBPField.COREMAP_VERSION.fieldName());
      if (coreMapVersion == null) {
        LuceneQuerier.logger.warn("no coremap version specified for document: " + doc.toString());
      }
      BytesRef data = doc.getBinaryValue(KBPField.COREMAP.fieldName());
      InputStream coremapStream;
      if (data != null) {
        byte[] coremapData = data.bytes;
        coremapStream = new ByteArrayInputStream(coremapData);
      } else {
        String coreMapFile = postProcessFilename(doc.get(KBPField.COREMAP_FILE.fieldName()));
        if (coreMapFile != null && new File(coreMapFile).exists() && new File(coreMapFile).canRead()) {
          coremapStream = new BufferedInputStream(new FileInputStream(coreMapFile));
        } else { return Maybe.Nothing(); }
      }
      try {
        Pair<Annotation, InputStream> pair = serializer.read(coremapStream);
        pair.second.close();
        Maybe<Annotation> annotation = Maybe.Just(pair.first);
        coremapStream.close();
        return annotation;
      } catch (KryoException e){
        LuceneQuerier.logger.err(e);
        return Maybe.Nothing();
      } catch (NullPointerException e){ // Protobufs cry null pointers
        LuceneQuerier.logger.err(e);
        return Maybe.Nothing();
      }
    }

    /**
     * In rare cases, we would like to load the annotations from a different path (e.g., one which is
     * cached on local disk); this function handles that remapping.
     * @param rawFilename The original filename to modify
     * @return The tweaked filename, usually with some part of the path changed
     */
    protected String postProcessFilename(String rawFilename) {
      String rewritten = rawFilename;
      for (Map.Entry<String, String> entry : Props.INDEX_READDOC_REWRITE.entrySet()) {
        rewritten = rewritten.replaceAll(entry.getKey(), entry.getValue());
        if (rewritten.equals(rawFilename)) {
          LuceneQuerier.logger.warn("rewrite pattern didn't apply: text='" + rawFilename + "' pattern=s/" + entry.getKey() + "/" + entry.getValue() + "/g");
        }
      }
      return rewritten;
    }
  }

}
