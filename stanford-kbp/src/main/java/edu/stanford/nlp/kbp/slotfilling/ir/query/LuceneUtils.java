package edu.stanford.nlp.kbp.slotfilling.ir.query;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexableField;

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Utility functions for working with Lucene
 *
 * @author Angel Chang
 */
public class LuceneUtils {
  // Checks if the reader has a particular field
  public static boolean hasField(IndexReader reader, String field) throws IOException{
    for (int i = 0; i < reader.maxDoc(); ++i) {
      Document candidate = reader.document(i);
      if (candidate != null) {
        return candidate.getField(field) != null;
      }
    }
    throw new IOException("Reader has no valid documents: " + reader);
  }

  // Returns a list of stored fields in the index (or at least the first document of the index)
  public static Set<String> getStoredFields(IndexReader reader) throws IOException {
    for (int i = 0; i < reader.maxDoc(); ++i) {
      Document candidate = reader.document(i);
      if (candidate != null) {
        List<IndexableField> fields = candidate.getFields();
        Set<String> fieldNames = new HashSet<String>();
        for (IndexableField field:fields) {
          fieldNames.add(field.name());
        }
        return fieldNames;
      }
    }
    throw new IOException("Reader has no valid documents: " + reader);
  }




}
