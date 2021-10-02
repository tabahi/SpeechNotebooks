const label_types = {cat: "cats", ord: "ords"};
var models_cats = null; //models object models_cats[db_id][label_name]
var models_ords = null;
let db_id = 0; //might use it later

let SPEC_BOX_WIDTH = 300;
let SPEC_BOX_HEIGHT = 200;

function start_nn_training(label_type, label_name)
{
    // Step 1: load data or create some data 
    //let label_name = 'A';
    //console.log(document.getElementById('text_NN_options_' + label_name).value);
    
    let options = null;
    try
    {
        options = JSON.parse(document.getElementById('text_NN_options_' + label_type +'_'+ label_name).value);
    }
    catch(e){ alert(e); return;}
    if(isObject(options)==false) {alert("Invalid syntax"); return}
    document.getElementById('nn_train_modal_' + label_type +'_'+ label_name).style.display='none';
    
    
    let epochs_ = parseInt(document.getElementById('epochs_' + label_type +'_'+ label_name).value); if(epochs_ < 1) epochs_ = 10;
    let batchSize_ = parseInt(document.getElementById('batch_' + label_type +'_'+ label_name).value); if(batchSize_ < 1) batchSize_ = 10;

    document.getElementById('nn_msg').textContent = "Processing data for training for label " + label_name;
    
    let train_data = collect_training_data(label_type, label_name);

    if(train_data)
    {
        const balance = true;
        // Step 2: set your neural network options
        
        console.log("Training " + label_type + "\t" + label_name);
        console.log(options);
        const trainingOptions = { epochs: epochs_ , batchSize: batchSize_ };
        console.log(trainingOptions);
        
        // Step 3: initialize your neural network
        //nn_model = null;
        let new_nn_model = ml5.neuralNetwork(options);

        // Step 4: add train_data to the neural network
        let data_len = train_data[0].length;
        let sample_count = 0;
        
        
        if((label_type==label_types.cat) && (label_heads_cat.length>0))
        {
            
            let label_ = null; let label_classes = [];
            for (let y_cat=0; y_cat < label_heads_cat.length; y_cat++)
            if(Object.keys(label_heads_cat[y_cat])[0] == label_name)
            {
                label_ = label_name;
                label_classes = label_heads_cat[y_cat][label_];
                break;
            }
            console.log(label_  +'\t' + label_classes);
            
            let count_n = new Array(label_classes.length).fill(0);
            var SpecCtx = document.getElementById('SpectrumCanvas').getContext('2d');

            function add_data(data_i, only_this_label=null)
            {
                if( train_data[1][data_i] && (isObject(train_data[1][data_i][0])) && (train_data[1][data_i][0][label_] != null) )
                {
                    const label_i = label_classes.indexOf(String(train_data[1][data_i][0][label_]));
                    
                    if( ((only_this_label==null)||(label_i==only_this_label))  && ((label_i>=0)||(label_classes.indexOf('*')>=0)) )
                    {
                        let this_out = train_data[1][data_i][0][label_name];
                        
                        const output = {label: String(this_out)};   //doesn't work without String
                        SpecCtx.clearRect(0, 0, SPEC_BOX_WIDTH, SPEC_BOX_HEIGHT);
                        SpecCtx.drawImage(document.getElementById(train_data[0][data_i]), 0, 0);
                        const input = SpecCtx.getImageData(0, 0, SPEC_BOX_WIDTH, SPEC_BOX_HEIGHT).data;
                        
                        new_nn_model.addData(input, output);

                        if(label_i>=0) count_n[label_i]++;

                        if(only_this_label==null)// before balancing
                            sample_count++;
                    }
                }
            }

            for (let i=0; i<data_len; i++)  add_data(i);
            console.log("Samples: " + String(count_n));
            

            if(sample_count<10) 
            {
                document.getElementById('nn_msg').textContent = "Sample size "+String(sample_count)+"/"+ String(data_len)+" too small for training";
                return;
            }
            
            if(balance)
            {
                const max_n = arrayMax(count_n);
                if(max_n > 0)
                for(let cl=0; cl < label_classes.length; cl++)
                {
                    while((count_n[cl]<max_n) && (count_n[cl] > 3))
                    for (let i=0; i<data_len; i++) 
                    {
                        if(count_n[cl]<max_n) add_data(i, cl);
                        if(count_n[cl]>=max_n) break;
                    }
                }
                console.log("Balanced:"); console.log(count_n);
            }
            

        }
        else if((label_type==label_types.ord) && (label_heads_ord.length>0))
        {
            let label_ = null;
            for (let y_ord=0; y_ord<label_heads_ord.length; y_ord++)
            if(label_heads_ord[y_ord] == label_name)
            {
                label_ = label_name;
                break;
            }

            alert("Ordinal labels CNN isn't coded yet. It's under construction. Sorry.");
            return;
        }
        else
        {
            console.error("Invalid label type");
            return;
        }
        
        
        document.getElementById('nn_msg').innerHTML = `Using ${sample_count}/${data_len} samples to train the model<br>`;
        document.getElementById('nn_msg').innerHTML += `<div class="w3-animate-fading"><b>Starting training for ${label_name}...</b></div>`;
        
        // Step 5: normalize your train_data;
        new_nn_model.normalizeData();
        
        
        // Step 6: train your neural network
        //const trainingOptions = { epochs: 32, batchSize: 12 };

        try
        {
            new_nn_model.train(trainingOptions, whileTraining, finishedTraining);
        }
        catch(e)
        {
            console.error(e);
            document.getElementById('nn_msg').innerText = e;
            new_nn_model = null; //error in training
        }
        

        function whileTraining(epoch, loss) {
            
            document.getElementById('nn_msg').textContent = "Epoch: " + epoch + ", loss: " + loss.loss.toFixed(3) + ", accuracy: " + loss.acc.toFixed(3);
        }
        
        // Step 7: use the trained model
        function finishedTraining()
        {
            
            //new_nn_model.save("model");
            if(label_type==label_types.cat)
            {
                if(!models_cats) models_cats = [];
                if(!models_cats[db_id]) models_cats[db_id] = {};
                models_cats[db_id][label_name] = null;
                models_cats[db_id][label_name] = new_nn_model;

                document.getElementById('nn_msg').textContent = "Finished training (c). New model is loaded for label: " + label_name;
                document.getElementById('pred_' + label_type + '_' + label_name).disabled = false;
            }
            else if(label_type==label_types.ord)
            {
                if(!models_ords) models_ords = [];
                if(!models_ords[db_id]) models_ords[db_id] = {};
                models_ords[db_id][label_name] = null;
                models_ords[db_id][label_name] = new_nn_model;

                document.getElementById('nn_msg').textContent = "Finished training (o). New model is loaded for label: " + label_name;

                document.getElementById('pred_' + label_type + '_' + label_name).disabled = false;
            }
        }
        
        train_data = null;
    }
    else
    {
        document.getElementById('nn_msg').textContent = "No data for training";
    }
}




function make_ML_div(db_id=0, enable_disable=null) //called from index and loading functions
{
    let ML_enable = true;
    if(enable_disable) ML_enable = enable_disable;

    if(ML_enable)
    {
        document.getElementById("ML_training_div").innerHTML = `<h3 class="w3-opacity">ML Training</h3>`;
        let thtml_b = "";      //html string for div box
        let thtml_m = "";   //html string for modals
        
        if(label_heads_cat)
        {
            for (let y_cat=0; y_cat < label_heads_cat.length; y_cat++)
            {
                const label_type = label_types.cat;
                
                if(isObject(label_heads_cat[y_cat]))
                {
                    const label_name = Object.keys(label_heads_cat[y_cat])[0];
                    let model_options = "{}";
                    if(models_cats && models_cats[db_id] && models_cats[db_id][label_name]) model_options = JSON.stringify(models_cats[db_id][label_name].options);
                    else model_options = nn_default_options_cats;
                    
                    //use default options when model is not loaded or available on server

                    thtml_b += build_ML_box(db_id, label_type, label_name, 'lime');
                    thtml_m += build_ML_modal(db_id, label_type, label_name, model_options, nn_default_options_cats);
                }
            }
        }

        if(label_heads_ord)
        {
            for (let y_ord=0; y_ord<label_heads_ord.length; y_ord++)
            {
                const label_type = label_types.ord;
                
                if(label_heads_ord[y_ord]!=null)
                {
                    const label_name = label_heads_ord[y_ord];
                    
                    let model_options = "{}";
                    if(models_ords && models_ords[db_id] && models_ords[db_id][label_name]) model_options = JSON.stringify(models_ords[db_id][label_name].options);
                    else model_options = nn_default_options_ords;

                    thtml_b += build_ML_box(db_id, label_type, label_name, 'indigo');
                    thtml_m += build_ML_modal(db_id, label_type, label_name, model_options, nn_default_options_ords);
                }
            }
            document.getElementById("ML_training_div").innerHTML += thtml_b;
            document.getElementById("ML_training_div_modals").innerHTML = thtml_m;
            thtml_b = null;
            thtml_m = null;
        }

        if(label_heads_cat)
        {
            //pre-load models - cats
            for (let y_cat=0; y_cat < label_heads_cat.length; y_cat++)
            {
                const label_type = label_types.cat;
                if(isObject(label_heads_cat[y_cat]))
                {
                    const label_name = Object.keys(label_heads_cat[y_cat])[0];
                    if(label_name=='emotion_xxxxx')   //only for emotion - debug
                    {
                        const model_link = models_dir+db_id+'/'+label_type+'_'+label_name+'/model.json';
                        load_single_nn(db_id, label_type, label_name, false, null, model_link).then(function()
                        {

                        }).catch((err)=>{
                        
                            console.error(err);
                            document.getElementById('nn_msg').textContent = err;
                            
                        });
                    }
                }
            }
        }

        if(label_heads_ord)
        {
            //pre-load models - ords
            for (let y_ord=0; y_ord<label_heads_ord.length; y_ord++)
            {
                const label_type = label_types.ord;
                
                if(label_heads_ord[y_ord]!=null)
                {
                    const label_name = label_heads_ord[y_ord];
                    
                    if(label_name=='V_xxxx')   //only for V - debug
                    {
                        const model_link = models_dir+db_id+'/'+label_type+'_'+label_name+'/model.json';
                        load_single_nn(db_id, label_type, label_name, false, null, model_link).then(function()
                        {

                        }).catch((err)=>{
                        
                            console.error(err);
                            document.getElementById('nn_msg').textContent = err;
                        });
                    }
                }
            }

        }
    }
    else
    {
        document.getElementById("ML_training_div").innerHTML = "";
    }
}


function isObject(objValue) {
    return objValue && typeof objValue === 'object' && objValue.constructor === Object;
}

function arrayMax(arr) {
    let len = arr.length, max = -Infinity;
    while (len--) {
      if (arr[len] > max) {
        max = arr[len];
      }
    }
    return max;
};

function build_ML_box(db_id, label_type, label_name, border_color='gray')
{
    return `<div class="w3-cell w3-border w3-border-${border_color}" style="width:100px; margin-left:10px; display: inline-block;">
                <h5>${label_name}</h5>
                <div class="w3-row w3-margin-top">
                <button onclick="document.getElementById('nn_train_modal_${label_type}_${label_name}').style.display='block'" class="w3-button w3-tiny w3-border w3-border-indigo">Model</button>
                </div>
                <div class="w3-row w3-margin-top">
                    <button onclick="start_nn_prediction('${label_type}', '${label_name}')" class="w3-button w3-tiny w3-border w3-border-indigo w3-margin-bottom" id="pred_${label_type}_${label_name}" disabled>Predict</button>
                </div>
            </div>`;
}

function build_ML_modal(db_id, label_type, label_name, model_options, default_options)
{
    
    //Train settings modals
    return `<div id="nn_train_modal_${label_type}_${label_name}" class="w3-modal w3-animate-opacity">
            <div class="w3-modal-content w3-card-4 w3-animate-zoom">

                <header class="w3-container w3-theme-d1"> 
                <span onclick="document.getElementById('nn_train_modal_${label_type}_${label_name}').style.display='none'" 
                class="w3-button w3-red w3-xlarge w3-display-topright">&times;</span>
                <h2>${label_type} CNN model for ${label_name}</h2>
                </header>

                <div class="w3-container w3-theme-d2 w3-small w3-left-align">
                    <div class="w3-row w3-padding-small ">
                        
                        <p>Download the trained model or load a pre-trained model</p>
                            
                        <div class="w3-col m2">
                            <button onclick="download_nn_model('${label_type}','${label_name}')" class="w3-button w3-tiny w3-border w3-indigo">Download Model</button>
                        </div>

                        <div class="w3-col m2">
                        <input type="file" id="nn_mod${label_type}_${label_name}" name="nn_${label_type}_${label_name}[]" onchange="load_files_nn_model(this.files, '${label_type}', '${label_name}')" multiple="" class="w3-button w3-hover-border-khaki w3-padding"/>

                        </div>
                    </div>
                    <h6>Model training options</h6>

                    <button onclick="document.getElementById('text_NN_options_${label_type}_${label_name}').value=unescape('${escape(default_options)}')" class="w3-button w3-tiny">Default Options</button>

                    <a href="https://learn.ml5js.org/#/reference/neural-network?id=defining-custom-layers"> References </a> 

                    <div class="w3-row w3-left-align">
                        <textarea name="text_NN_options_${label_type}_${label_name}" id="text_NN_options_${label_type}_${label_name}" cols="10" rows="25" class="w3-input w3-dark-input" style="font-family: 'Courier New', Courier, monospace; max-height:300px;">${model_options}</textarea>
                    </div>
                </div>
                

                <div class="w3-row-padding w3-theme-d2 w3-left-align w3-small">
                    <div class="w3-third">
                        <label for="epochs">Epochs</label>
                        <input class="w3-input w3-dark-input w3-margin-right" type="number" id="epochs_${label_type}_${label_name}" placeholder="10" value="10">
                    </div>
                    <div class="w3-third">
                        <label for="batch">Batch size</label>
                        <input class="w3-input w3-dark-input" type="number" id="batch_${label_type}_${label_name}" placeholder="10" value="10">
                    </div>
                    
                    <div class="w3-third">
                    <span>DB ID: ${db_id}</span>
                    </div>
                </div> 

                <div class="w3-container  w3-theme-d1 w3-padding">

                <button class="w3-button w3-left w3-blue-grey w3-border w3-round-large w3-margin-left" onclick="document.getElementById('nn_train_modal_${label_type}_${label_name}').style.display='none'">Close</button>

                    <button class="w3-button w3-left w3-teal w3-border w3-round-large w3-margin-right"
                    onclick="start_nn_training('${label_type}', '${label_name}')">Start Training</button>

                </div>
            </div>
        </div>`;
}


const nn_default_options_cats = `{
    "task": "imageClassification",
    "inputs":[${SPEC_BOX_WIDTH}, ${SPEC_BOX_HEIGHT}, 4],
    "outputs": 1,
    "debug": true,
    "layers":
    [
        {
            "type": "conv2d",
            "filters": 8,
            "kernelSize": 5,
            "strides": 1,
            "activation": "relu",
            "kernelInitializer": "varianceScaling"
        },
        {
            "type": "maxPooling2d",
            "poolSize": [2, 2],
            "strides": [2, 2]
        },
        {
            "type": "conv2d",
            "filters": 16,
            "kernelSize": 5,
            "strides": 1,
            "activation": "relu",
            "kernelInitializer": "varianceScaling"
        },
        {
            "type": "maxPooling2d",
            "poolSize": [2, 2],
            "strides": [2, 2]
        },
        {
            "type": "flatten"
        },
        {
            "type": "dense",
            "kernelInitializer": "varianceScaling",
            "activation": "softmax"
        }
    ]
}`;


const nn_default_options_ords = `{
    "task": "regression",
    "inputs": 1,
    "outputs": 1,
    "learningRate": 0.2,
    "debug": true,
    "layers":
    [
        {
            "type": "dense",
            "units": 64,
            "activation": "sigmoid"
        },
        {
            "type": "dense",
            "units": 16,
            "activation": "sigmoid"
        },
        {
            "type": "dense",
            "activation": "sigmoid"
        }
    ]
}`;

