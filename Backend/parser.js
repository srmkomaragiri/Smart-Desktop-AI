/**
 * parser.js
 * Parses the highly structured output from the Python AI engine into JSON.
 */

function parseEngineOutput(rawText) {
    const result = {
        raw_text: "",
        valid_code: "",
        noise: "",
        issues: [],
        rag_insights: [],
        explanation: "",
        suggestions: [],
        confidence: {
            text: 0,
            explanation: 0
        }
    };

    if (!rawText) return result;

    // Helper to extract a section block between a regex header and the next number header or end of file
    const extractSection = (regexStr, isList = false) => {
        const regex = new RegExp(`(?:\\d\\.\\s+)?${regexStr}[\\s\\S]*?(?=(?:\\n\\d\\.\\s+[A-Z])|$)`, 'i');
        const match = rawText.match(regex);
        if (!match) return isList ? [] : "";
        
        let block = match[0].replace(new RegExp(`^(?:\\d\\.\\s+)?${regexStr}\\s*\\n?`, 'i'), '').trim();
        
        if (isList) {
            // Split by list bullets or line breaks
            return block.split('\n').map(l => l.replace(/^[\*\-]\s*/, '').trim()).filter(l => l);
        }
        return block;
    };

    result.raw_text = extractSection("RAW EXTRACTED TEXT");
    result.noise = extractSection("NOISE(?:[\\s\\S]*?if any\\))?");
    result.valid_code = extractSection("VALID CODE");
    result.issues = extractSection("ISSUES DETECTED", true);
    result.rag_insights = extractSection("RAG INSIGHTS", true);
    result.explanation = extractSection("EXPLANATION");

    // Extract MULTIPLE CORRECT SUGGESTIONS
    // Match block and extract numbered items
    const suggBlockMatch = rawText.match(/(?:7\.\s+)?MULTIPLE CORRECT SUGGESTIONS\s*([\s\S]*?)(?:(?=\n\d\.\s+[A-Z])|$)/i);
    if (suggBlockMatch) {
        let text = suggBlockMatch[1];
        const itemRegex = /^\s*(\d+)\.\s+(.*?)(?=\n\s*\d+\.|\Z)/gms;
        let m;
        while ((m = itemRegex.exec(text)) !== null) {
            let id = parseInt(m[1], 10);
            let content = m[2].trim();
            // Split into title (first line) and code (rest)
            let lines = content.split('\n');
            let title = lines[0].trim();
            let code = lines.slice(1).join('\n').trim();
            result.suggestions.push({
                id: id,
                title: title.replace(/^If you want to /i, ''),
                code: code
            });
        }
    }

    // Extract CONFIDENCE REPORT
    const confBlockMatch = rawText.match(/(?:8\.\s+)?CONFIDENCE REPORT\s*([\s\S]*?)$/i);
    if (confBlockMatch) {
        let text = confBlockMatch[1];
        const screenConfMatch = text.match(/Exact Text on Screen:\s*(\d+)%/i);
        const explConfMatch = text.match(/Explanation Accuracy:\s*(\d+)%/i);
        
        if (screenConfMatch) result.confidence.text = parseInt(screenConfMatch[1], 10);
        if (explConfMatch) result.confidence.explanation = parseInt(explConfMatch[1], 10);
    }

    return result;
}

module.exports = { parseEngineOutput };
