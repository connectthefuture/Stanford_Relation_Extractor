package edu.stanford.nlp.kbp.slotfilling.ir.query;

import edu.stanford.nlp.kbp.common.*;
import edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet;
import edu.stanford.nlp.kbp.slotfilling.ir.index.KBPField;
import edu.stanford.nlp.util.IterableIterator;
import edu.stanford.nlp.util.Pair;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import org.apache.lucene.store.FSDirectory;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * A querier which always returns the correct documents, as judged by
 * the correct answers to the task.
 *
 * @author Gabor Angeli
 */
public class LuceneGoldQuerier extends LuceneQuerier {

  public final GoldResponseSet goldProvenances;

  public LuceneGoldQuerier(IndexReader reader, GoldResponseSet goldProvenances) {
    super(reader);
    this.goldProvenances = goldProvenances;
  }

  public LuceneGoldQuerier(File reader, GoldResponseSet goldProvenances) throws IOException {
    this(DirectoryReader.open(FSDirectory.open(reader)), goldProvenances);
  }

  @Override
  protected IterableIterator<Pair<Integer, Double>> queryImplementation(String entityName, Maybe<NERTag> entityType, Maybe<String> relation, Maybe<String> slotFill, Maybe<NERTag> slotFillType, Maybe<Integer> maxDocuments) throws IOException {
    // Find gold responses
    final Set<String> gold = new HashSet<String>();
    for (Map.Entry<KBPSlotFill, Set<String>> possibleKey : goldProvenances.correctProvenances().entrySet()) {
      boolean match = true;
      if (!entityName.toLowerCase().trim().equals(possibleKey.getKey().key.entityName.toLowerCase().trim())) {
        match = false;
      }
      for (String rel : relation) {
        if (!rel.toLowerCase().trim().equals(possibleKey.getKey().key.relationName.toLowerCase().trim())) {
          match = false;
        }
      }
      for (String fill : slotFill) {
        if (!fill.toLowerCase().trim().equals(possibleKey.getKey().key.slotValue.toLowerCase().trim())) {
          match = false;
        }
      }
      if (match) {
        gold.addAll(possibleKey.getValue());
      }
    }

    return new IterableIterator<Pair<Integer, Double>>(new Iterator<Pair<Integer, Double>>() {
      private Iterator<String> impl = gold.iterator();
      private String nextDocId = null;
      @Override
      public boolean hasNext() {
        while (nextDocId == null && impl.hasNext()) {
          nextDocId = impl.next();
          if (nextDocId.toLowerCase().startsWith("noid")) { nextDocId = null; }
        }
        return nextDocId != null;
      }
      @Override
      public Pair<Integer, Double> next() {
        if (!hasNext()) { throw new NoSuchElementException(); }
        Pair<Integer, Double> toReturn = docIdToLuceneId(nextDocId);
        nextDocId = null;
        return toReturn;
      }
      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }
    });
  }


  private Pair<Integer,Double> docIdToLuceneId(String docId) {
    try {
      TopDocs results = this.searcher.search(new TermQuery(new Term(KBPField.DOCID.fieldName(), docId)), 1);
      if (results.scoreDocs.length == 0) {
        throw new IllegalArgumentException("No such docid: " + docId);
      }
      return Pair.makePair(results.scoreDocs[0].doc, 1.0);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
