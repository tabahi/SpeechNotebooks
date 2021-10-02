
var loaded_labels = null;
let json_loaded = false;
var loaded_spectra = null;


var label_heads_cat = [];
var label_heads_ord = [];


function readmultifiles(files) 
{
    loaded_spectra = null;
    loaded_spectra = files;
    document.getElementById('nn_msg').textContent = "Total " + String(loaded_spectra.length) + " files loaded";

    refresh_db_table();
}

function collect_training_data()
{
    let files_spectra = [];
    let files_true_labels = [];
    let files_pred_labels = [];
    for (let i=0; i<loaded_spectra.length; i++)
    {
        let name_split = loaded_spectra[i].name.split('~');
        let file_label = label_from_filename(name_split[0]);
        
        //let htmlImgElm = document.getElementById(name_split[0]);

        if((name_split[0]) && (file_label))
        {
            files_spectra.push(name_split[0]);
            files_true_labels.push(file_label);
            files_pred_labels.push(null);
        }
    }
    
    return [files_spectra, files_true_labels, files_pred_labels];
}


function refresh_db_table()
{
    check_label_heads();
    let thtml = `<table id="table_0" class="center w3-table w3-bordered">
    <tr>
    <th>File</th>
    <th>Spectrum</th>`;
    for (let y_cat=0; y_cat < label_heads_cat.length; y_cat++)
        thtml += `<th><i>${Object.keys(label_heads_cat[y_cat])[0]}</i></th>`;

    for (let y_ord=0; y_ord<label_heads_ord.length; y_ord++)
        thtml += `<th><i>${label_heads_ord[y_ord]}</i></th>`;

    thtml += `</tr>`;

    for (let i=0; i<loaded_spectra.length; i++)
    {
        let name_split = loaded_spectra[i].name.split('~');
        let file_label = label_from_filename(name_split[0]);
        thtml += `<tr><td>${name_split[0]}</td>`;
        thtml += `<td><img id=${name_split[0]} src="${URL.createObjectURL(loaded_spectra[i])}" width="200" height="40"></td>`;

        
        if(file_label)
        for (let y_cat=0; y_cat < label_heads_cat.length; y_cat++)
        {
            const label_ = Object.keys(label_heads_cat[y_cat])[0];
            thtml += `<td><i>${(file_label[0][label_]) ? file_label[0][label_]: null}</i></td>`;
        }
        
        if(file_label)
        for (let y_ord=0; y_ord<label_heads_ord.length; y_ord++)
        {
            const label_ = label_heads_ord[y_ord];
            thtml += `<td><i>${(file_label[1][label_]) ? file_label[1][label_] : null}</i></td>`;
        }
        thtml += `</tr>`;
    }
    document.getElementById("table_div").innerHTML = thtml;
    if(label_heads_cat.length > 0 || label_heads_ord.length > 0)
    {
        make_ML_div();
    }

}


function handleFileSelect_labels(e) 
{

    let uploaded_file = e.target.files[0];

    var reader = new FileReader();

    // Closure to capture the file information.
    reader.onload = (function(theFile) {
        return function(e) {
            
            Load_JSON_Labels_file(e.target.result);
            refresh_db_table();
            
        };
    })(uploaded_file);

    // Read in the image file as a data URL.
    reader.readAsText(uploaded_file);

}

function Load_JSON_Labels_file(json_string)  //seg_key, key_ts, key_true, key_pred
{
    
    let new_data = JSON.parse(json_string);
    if(new_data)
    {
        if(new_data[0] && new_data[0].i)
        {
        loaded_labels = new_data
        const labels_n = loaded_labels.length;
        json_loaded = true;
        document.getElementById('nn_msg').textContent = "Loaded " + String(labels_n)  + " labels from JSON";
        }
        else
        {
            alert("Invalid label file");
        }
    }
    return;
}


function label_from_filename(filename) //called from localstore
{
    //custom function to extract labels from filename
    //specifically codes for emotion recognition, extracts labels of emo, A, V,D, and sex from loaded_labels (a formantted Json file)

    if(filename)
    {
        if(json_loaded)
        {
            let this_name = filename;
            for(let i=0; i<loaded_labels.length;i++)
            {
                if(loaded_labels[i].i==this_name)
                {
                    const label_cats = { emotion: loaded_labels[i].emo, sex: loaded_labels[i].sex, spkr: loaded_labels[i].spkr , U: loaded_labels[i].U };
                    const label_ords = { V: loaded_labels[i].V, A:loaded_labels[i].A , D: loaded_labels[i].D };
                    
                    return [label_cats, label_ords];
                }
            }
            console.log('Not found ' + this_name);
            return null;
        }
        else
        {
            //console.warn("Invalid filename, can't extract label from it.\t"+(filename));
            return null;
        }
    }
    else
    {
        console.error("Error: Filename is required to extract emo labels.");
        return null;
    }
}


function check_label_heads()
{
    try
    {
        let xx43 = document.getElementById('class_labels').value;
        if(xx43) xx43 = JSON.parse(document.getElementById('class_labels').value);
        if(xx43!==null) { if (xx43.length==0) label_heads_cat = []; else label_heads_cat = xx43; }
        let xx44 = document.getElementById('ordinal_labels').value;
        if(xx44) xx44 = JSON.parse(xx44);
        if(xx44!==null) { if (xx44.length==0) label_heads_ord = []; else label_heads_ord = xx44;}
    }
    catch(e)
    {
        alert(e);
    }
}



document.getElementById("up_labels_file").addEventListener("change", handleFileSelect_labels, false);
