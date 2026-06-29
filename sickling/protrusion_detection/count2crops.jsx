#target photoshop;
main();
function main(){
var ref = new ActionReference();
ref.putProperty(charIDToTypeID('Prpr'),stringIDToTypeID("countClass"));
ref.putEnumerated( charIDToTypeID("Dcmn"), charIDToTypeID("Ordn"), charIDToTypeID("Trgt") ); 
var Count = executeActionGet(ref).getList(stringIDToTypeID("countClass")).count
for(var z=0; z < Count; z++){
var X = executeActionGet(ref).getList(stringIDToTypeID("countClass")).getObjectValue(z).getUnitDoubleValue(stringIDToTypeID( "x" ));
var Y = executeActionGet(ref).getList(stringIDToTypeID("countClass")).getObjectValue(z).getUnitDoubleValue(stringIDToTypeID( "y" ));
var layerName = (z+1);
SaveSelection(X,Y,layerName); 
}
};

function SaveSelection(X,Y,layerName){
	var docRef = activeDocument;
	var fileNameNoExtension = docRef.name;
	var thisFile = new File($.fileName);  
	var basePath = thisFile.path;
	// specify crop image dimensions
	var sw = 30;
	var sh = 30;
	var bounds = [ [X-sw/2,Y-sh/2], [X-sw/2,Y+sh/2], [X+sw/2,Y+sh/2], [X+sw/2,Y-sh/2] ];
	app.activeDocument.selection.select(bounds);
	app.activeDocument.selection.copy();
	app.preferences.rulerUnits = Units.PIXELS
    docRef = app.documents.add(sw,sh);
    docRef.paste();
	
	var saveOptions = new JPEGSaveOptions( );  
	saveOptions.embedColorProfile = true;  
	saveOptions.formatOptions = FormatOptions.STANDARDBASELINE;  
	saveOptions.matte = MatteType.NONE;  
	saveOptions.quality = 12;
	var saveFile = new File(basePath + "/" + layerName + ".jpg")
	app.activeDocument.saveAs( saveFile, saveOptions, true );
	
	app.activeDocument.close(SaveOptions.DONOTSAVECHANGES);
	
	}