package edu.stanford.nlp.kbp.common;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.util.CoreMap;

public class SentenceTriple {
	public String sentence;
	public String relationType;
	public KBPRelationProvenance provenance;
	
	public SentenceTriple(String rt, String sent, String p){
		sentence=sent;
		relationType=rt;
		
		//process provenance string to split different info
		String delimiter=":";
		String[] fields=p.split(delimiter);
		
		
		String docId=fields[0];
		String[] emspan= fields[4].split("-");		
		Span entitymention=new Span(Integer.parseInt(emspan[0]),Integer.parseInt(emspan[1]));
		
		String[] svspan=fields[3].split("-");
		Span slotvalue=new Span(Integer.parseInt(svspan[0]),Integer.parseInt(svspan[1]));
		Integer sentenceIdx = Integer.parseInt(fields[2]);

		//String[] justspan=fields[5].split("-");
		//Span justification=new Span(Integer.parseInt(justspan[0]),Integer.parseInt(justspan[1]));
		
		provenance=new KBPRelationProvenance(docId, null, sentenceIdx, entitymention, slotvalue,null, Maybe.Just(0.0));
		
	
	}

}