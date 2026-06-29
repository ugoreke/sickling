// count2csv.jsx — Photoshop script: write Count-tool marker coordinates to CSV.
//
// Variation of count2crops.jsx. Instead of cropping a small image around
// each count marker, this writes one (x, y) per row to <doc>_counts.csv
// in the same folder as the active document, in placement order.
//
// Convention used by the polymer-length tooling (see
// sickling/notebooks/polymer_length_grid.ipynb):
//   - rows 1, 3, 5, ...   = start coordinate of a polymer
//   - rows 2, 4, 6, ...   = end coordinate of the same polymer
//
// Drop two count markers per polymer (start + end). The downstream tool
// pairs consecutive rows. Order matters — place start first, then end.

#target photoshop;
main();

function main() {
    var ref = new ActionReference();
    ref.putProperty(charIDToTypeID('Prpr'), stringIDToTypeID("countClass"));
    ref.putEnumerated(charIDToTypeID("Dcmn"), charIDToTypeID("Ordn"), charIDToTypeID("Trgt"));

    var counts = executeActionGet(ref).getList(stringIDToTypeID("countClass"));
    var n = counts.count;
    if (n === 0) {
        alert("No count markers on the active document.");
        return;
    }

    var docRef = activeDocument;
    var thisFile = new File($.fileName);
    var basePath = thisFile.path;

    // Strip the file extension from the document name for the CSV name.
    var docName = docRef.name.replace(/\.[^\/.]+$/, "");
    var outFile = new File(basePath + "/" + docName + "_counts.csv");

    outFile.encoding = "UTF8";
    outFile.open("w");
    outFile.writeln("idx,x,y");

    for (var z = 0; z < n; z++) {
        var obj = counts.getObjectValue(z);
        var X = obj.getUnitDoubleValue(stringIDToTypeID("x"));
        var Y = obj.getUnitDoubleValue(stringIDToTypeID("y"));
        outFile.writeln((z + 1) + "," + X + "," + Y);
    }
    outFile.close();

    alert("Wrote " + n + " count markers to:\n" + outFile.fsName);
}
