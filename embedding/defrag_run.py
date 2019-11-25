def defrag_run(path2data, W):
    
    Wa = W["Wa"]
    Wf = W["Wf"]
    
    __lr_reduce = get_param('lr_reduce', 1.0)
    __batch_size = params.embedding.train.batch_size
    __max_epochs = params.embedding.train.max_epochs
    params = setup_activation_fn(params)
    
    print("getting training fold...\n")
    train_split = get_dataset_fold('train', path2data)
    print("getting validation fold...\n")
    vald_split = get_dataset_fold("val", path2data)
    Ni = train_split["frames"].shape[0] #TODO: verify
    
    max_iters = np.ceil(Ni/__batch_size) * __max_epochs
    ppi = 1 #how often (in unit of epochs) to evaluate validation error and (maybe) save checkpoint
    pp = np.ceil(ppi*(Ni/__batch_size)) #convert ppi from units of epoochs to units of iterations
    
    #MAJOR #TODO: do separately for Wa and Wf
    Wa_Eg = np.zeros()#TODO
    Wf_Eg = #TODO
    raw_costs = np.zeros(max_iters,1)
    reg_costs = np.zeros(max_iters, 1)
    
    __best_score = get_param('min_save_score', -1)
    
    hist_iter = []
    hist_e2v  = [] #validation score
    hist_e3v  = []
    hist_e2t  = [] #training score
    hist_e3t  = []
    
#     Wa_best = [] #MAJOR #TODO: do separately for Wa and Wf
#     Wf_best = []
    W_best = {}
    print("Starting Optimisation")
    
    batches_indices = #TODO
    #to come up with more efficient and pythonic way of creating batches here
    #remember to handle the last incomplete batch properly; strategy1: fill gap in last with first n 
    
    for itr in range(max_iters):
        curr_batch_indices = batches_indices[itr]
        frames_batch = train_split["frames"][curr_batch_indices]
        labels_batch = train_split["labels"][curr_batch_indices]
        
        #evaluate cost and gradients. All magic happens inside!
        cost, W_grad = defrag_cost(frames_batch, labels_batch, W)
        Wa_grad = W_grad["Wa_gradient"]
        Wf_grad = W_grad["Wf_gradient"]
        
        #learning rate modulation
        __lrmod = 1
        if (itr+1)/max_iters > __lr_reduce
            __lrmod = 0.1
        
        #sgd momentum
        Wa_dx = params.embedding.train.momentum * Wa_Eg - __lrmod * params.embedding.train.lr * Wa_grad
        Wf_dx = params.embedding.train.momentum * Wf_Eg - __lrmod * params.embedding.train.lr * Wf_grad
        Wa_Eg = Wa_dx
        Wf_Eg = Wf_dx
        
        #parameter update
        Wa = Wa + Wa_dx
        Wf = Wf + Wf_dx
        
        #keeping track of costs
        print("iter %d/%d (%.1f%% done): raw_cost: %f \t reg_cost: %f \t total_cost: %f\n", itr+1 , max_iters, 100*itr/max_iters,
              cost["raw_cost"], cost["reg_cost"], cost["total_cost"])
        raw_costs[itr] = cost["raw_cost"]
        reg_costs[itr] = cost["reg_cost"]
        total_cost = cost["reg_cost"] + cost["raw_cost"]
        if (math.isnan(total_cost) ):
            print("WARNING! COST WAS NAN!!! ABORTING \n")
            break; #something blew up, get out
        
        if itr+1 ==1:
            score0 = total_cost #remember cost at beginning
        if total_cost > score0*10:
            print("WARNING! TOTAL COST SEEMS TO BE EXPLODING!!! ABORTING \n")
            break; #we're exploding: learning rate too high or something. get out
        
        #eval validation performance every now and then, or on final iteration
        if ( ((itr+1)%pp ==0 and (itr+1) < 0.99*max_iters) || itr+1 == max_iters):
            
            if not vald_split.frames :
                #eval and record validation set performance
                e2r,e3r = defrag_eval(vald_split, W)
            else :
                e2r = []
                e3r = []
            e2r = np.array(e2r)
            e3r = np.array(e3r)
            print('validation performance:\n');
            print('image search     : r@1: %.1f,\t r@5: %.1f,\t r@10: %.1f,\t rAVG: %.1f,\t rMED: %d\n', np.mean(e2r<=1)*100, np.mean(e2r<=5)*100,
                  np.mean(e2r<=10)*100, np.mean(e2r), np.floor(np.median(e2r)));
            print('image annotation : r@1: %.1f,\t r@5: %.1f,\t r@10: %.1f,\t rAVG: %.1f,\t rMED: %d\n', np.mean(e3r<=1)*100, np.mean(e3r<=5)*100,
                  np.mean(e3r<=10)*100, np.mean(e3r), np.floor(np.median(e3r)));
            print('-----\n');
            score = (np.mean(e2r<=10)*100 + np.mean(e3r<=10)*100)/2; #take average R@10 as score
            hist_iter.append(itr+1)
            hist_e2v.append(np.mean(e2r<=10)*100)
            hist_e3v.append(np.mean(e3r<=10)*100)
            
            #generating report summarising training info
            report{                
                "raw_costs" : raw_costs,
                "reg_costs" : reg_costs,
                "iter" : itr+1,
                "max_iters": max_iters
                "val_e2r" : e2r,
                "val_e3r" : e3r,
                "hist_iter" : hist_iter,
                "hist_e2v" : hist_e2v,
                "hist_e3v" : hist_e3v,
                "val_score" : score
            }
        
            #TODO #DOUBT why was the following line commented out in original matlab code
            #evaluate training set based on random subset of examples equal in size to the validation set
            e2r, e3r = DeFragEval(train_split, params, theta, decodeInfo)
            e2r = np.array(e2r)
            e3r = np.array(e3r)
            #e2r = [] ;  e3r = [];
            print('training performance:\n');
            print('image search     : r@1: %.1f,\t r@5: %.1f,\t r@10: %.1f,\t rAVG: %.1f,\t rMED: %d\n', np.mean(e2r<=1)*100, np.mean(e2r<=5)*100,
                  np.mean(e2r<=10)*100, np.mean(e2r), np.floor(np.median(e2r)));
            print('image annotation : r@1: %.1f,\t r@5: %.1f,\t r@10: %.1f,\t rAVG: %.1f,\t rMED: %d\n', np.mean(e3r<=1)*100, np.mean(e3r<=5)*100,
                  np.mean(e3r<=10)*100, np.mean(e3r), np.floor(np.median(e3r)));
            print('-----\n');
            score = (np.mean(e2r<=10)*100 + np.mean(e3r<=10)*100)/2; # take average R@10 as score
            hist_e2t.append( np.mean(e2r<=10)*100)
            hist_e3t.append( np.mean(e3r<=10)*100)
            
            save_params = params
            save_params.f =0
            save_params.df =0
            
            report.update({
                "params": save_params,
                "train_e2r" : e2r,
                "train_e3r" : e3r,
                "hist_e2t" : hist_e2t,
                "hist_e3t" : hist_e3t,
                "train_score" : score
            })
            
            if(itr+1 == max_iters):
                 #this is the last iteration. Lets save a record of how it went
                top_val_score = np.max(0.5*(np.array(hist_e2v) + np.array(hist_e3v)));
                top_train_score = np.max(0.5*(np.array(hist_e2t) + np.array(hist_e3t)));
                randnum = np.floor(np.rand()*10000);
        
            #check if the performance is best so far, and if so save results
            if report["val_score"] > best_score :
                best_score = report["val_score"]
                W_best = {
                    "Wa": Wa,
                    "Wf": Wf
                }
                #back up parameters into checkpoint
                report.update({
                    "W": W_best
                })
    if W_best:
        W_best = {"Wa": Wa,"Wf": Wf}
    
    return W_best
                
                
            
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
        
    
    