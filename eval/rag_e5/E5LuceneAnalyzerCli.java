import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.StringJoiner;

import io.anserini.analysis.DefaultEnglishAnalyzer;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

/** Tokenize base64 TSV input with Pyserini/Anserini's default English analyzer. */
public final class E5LuceneAnalyzerCli {
    private E5LuceneAnalyzerCli() {}

    public static void main(String[] args) throws Exception {
        try (Analyzer analyzer = DefaultEnglishAnalyzer.newStemmingInstance("porter");
             BufferedReader reader = new BufferedReader(
                 new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                int separator = line.indexOf('\t');
                if (separator < 1) {
                    throw new IllegalArgumentException("input row must be id<TAB>base64(text)");
                }
                String id = line.substring(0, separator);
                byte[] encoded = Base64.getDecoder().decode(line.substring(separator + 1));
                String text = new String(encoded, StandardCharsets.UTF_8);
                StringJoiner tokens = new StringJoiner(" ");
                try (TokenStream stream = analyzer.tokenStream("contents", text)) {
                    CharTermAttribute term = stream.addAttribute(CharTermAttribute.class);
                    stream.reset();
                    while (stream.incrementToken()) {
                        tokens.add(term.toString());
                    }
                    stream.end();
                }
                String row = id + "\t" + tokens;
                System.out.println(
                    Base64.getEncoder().encodeToString(row.getBytes(StandardCharsets.UTF_8))
                );
            }
        }
    }
}
