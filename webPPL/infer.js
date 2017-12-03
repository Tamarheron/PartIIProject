var max = function(a,b){
  return Math.max(a,b);
};

var createMDP = function(options){
  // takes parameters: states, startStates, actions, transitions, utilities, betas
  assert.ok(_.has(options, 'states') &&
            _.has(options, 'startStates') &&
            _.has(options, 'actions') &&
            _.has(options, 'transitions') &&
            _.has(options, 'utilities') &&
            _.has(options, 'betas'),
      'createMDP args: options does not contain one or more' +
            ' of states, startStates, actions, transitions, utilities, betas');
  var states = options.states;
  
  var getBeta = function(state){
    var index = states.indexOf(state);
    assert.ok((index!=-1), 'state not found');
    return options.betas[index];
  };
  
  var getActions = function(state){
    var index = options.states.indexOf(state);
    assert.ok((index!=-1), 'state '+state+' not found');
    return options.actions[index];
  };
  
  var getUtility = function(state){
    var index = options.states.indexOf(state);
    assert.ok((index!=-1), 'state not found');
    return options.utilities[index];
  };
  var transitions = options.transitions;
  var getTransition = options.transitions;
  var startStates = options.startStates;
  var length = options.length;
  var actions = options.actions;
  var utilities = options.utilities;
  var betas = options.betas;
  
  var getExpectedUtility = function(options) {
       assert.ok(_.has(options, 'state') &&
            _.has(options, 'action'), 
                 'getExpectedUtility args:' +
                 'one or more of state, action is missing');
    
    var expected =  expectation(Infer({method:"enumerate", model() { 
         var current = options.state;
         var action = options.action;
         var next = sample(getTransition({state:current, action:action}));
         var util = getUtility(next);
         return util;

       }}));     
    return expected;
    

  };
  
  return{getBeta, getActions, getUtility, getExpectedUtility,
         getTransition, startStates, states, actions, transitions,
        utilities, betas};
};

  


var softMaxAgent = function(options) {
  assert.ok( _.has(options, 'MDP') &&
            _.has(options, 'discount'),
                 'softMaxAgent args:' +
                 'one or more of discount, MDP is missing');
  var MDP = options.MDP;
  var getActions = MDP.getActions;
  var getBeta = MDP.getBeta;
  var discount = options.discount;
  var getTransition=MDP.getTransition;
  var getUtility = MDP.getUtility;
  var debug = false;
  var getEU = dp.cache(function(options) {
    
    if(debug){console.log('called eu');};

       assert.ok(_.has(options, 'state') &&
                 _.has(options, 'action') &&
                 _.has(options, 'length'), 
                 'getEU args:' +
                 'one or more of state, action is missing');

    var current = options.state;
    var action = options.action;
    var length = options.length;
    var currentUtil = getUtility(current);
    
    var discountedFutureUtil =  expectation(Infer({method:'enumerate', model() { 
       var next = sample(getTransition({state:current, action:action}));
       assert.ok((length>0), 'length cannot be negative');
       if (length==1){
          return 0;
       } else{
         var newLength = length-1;
         var nextAction = sample(act({state:next, length:newLength}));
         var futureUtil = getEU({
           state:next, action:nextAction, length:newLength})
         var discountedFutureUtil = discount*futureUtil;
         if(debug){
          console.log('current state: '+current+
            '   currentUtil: '+currentUtil+
            '  nextAction: '+nextAction+'   next state: '+next+
            '  length: '+length+
            '  dctdFutUtil: '+discountedFutureUtil);}
         return (discountedFutureUtil);
       }}})); 
    var total = currentUtil+discountedFutureUtil;
    if(debug){console.log('computed eu: '+total);}
    return total;
  });
  
  var act = dp.cache(function(options){
    //console.log('called act');
    assert.ok(_.has(options, 'state') &&
              _.has(options, 'length'), 
                'act args: missing one or more of: state, length');
    var state=options.state;
    var length=options.length;
    var beta = getBeta(state);
    var actions = getActions(state);
    return Infer({ method:"enumerate",
        model() {
          var action = uniformDraw(actions);
          var eu = getEU({state: state, action:action, length:length});
          var debugPrint = false;
          if (debugPrint){
            console.log("action: " + action + "   state: " + state + 
                  '  length: ' + length +
                 "   beta: " + beta + " eu: "+eu+"   factor: "+eu/beta );
          }
          factor(eu/beta);
          
        return action;
        }
   });
  });

  return {getEU, act};
};

//generate sample trajectory
var makeTrajectory = function(options) {
   assert.ok(_.has(options, 'length') &&
             _.has(options, 'agent') &&
            _.has(options, 'MDP'), 
                 'makeTrajectory args:' +
                 'one or more of length, agent, MDP is missing');
  var debug=false;
  if(debug){console.log('calling maketraj');}

  var MDP = options.MDP;
  var makeTrajectoryLength = options.length;
  var start = uniformDraw(MDP.startStates);
  
  if(debug){
  console.log('start is ' + start + 
    ' makeTrajectoryLength is '+ makeTrajectoryLength);
  }
  var agent = options.agent;
  var act=agent.act;
  var transition=MDP.getTransition;
  
  var step = function(options){
    //console.log('calling step');
    assert.ok(_.has(options, 'state') &&
              _.has(options, 'stepLength'), 
                 'makeTrajectory.step args:' +
                 'one or more of stepLength, state is missing');
    var state=options.state;
    var stepLength=options.stepLength;
    var action = sample(act({state:state, length:stepLength}));
    var transDist = transition({state:state, action:action})
    var next = sample(transDist);
    var res = [state, action];
    var newLength = stepLength-1;
    var debug=false;
    if (debug) {console.log('action: ' + action + ' trnsDist: '+transDist+
               ' stepLength: '+stepLength +' newLen: '+newLength+ ' next: '+ next);};
    return stepLength==1 ? 
    [res] : 
    [res].concat(step({state:next, stepLength:newLength}));
  };
  
  var result= step({state:start, stepLength:makeTrajectoryLength});
  return result;
  
};


var posterior = function(options){
  assert.ok(_.has(options, 'priors') &&
            _.has(options, 'MDPParams') &&
            _.has(options, 'discount') &&
            _.has(options, 'beta') &&
            _.has(options, 'observedTrajectory'),
           'posterior args missing one or more of ' +
            'priors, MDPParams,discount, beta (betaType), observedTrajectory');
  
  var betaType=options.beta;
  //assert.ok((betaType=='fixed') || (betaType=='variable'), 'beta must be "fixed"+
  //          ' or "variable"');
  
  var priors = options.priors;
  var traj = options.observedTrajectory;
  var params = options.MDPParams;
  var discount=options.discount;
  
  var trajLength = traj.length;
  var remainingLengths = mapN(
      function(n){return trajLength-n}, trajLength)
  
  var debug =false;
  
  return Infer({method:"enumerate",model(){
    //sample from priors
    var utilities = map(sample,priors.utilities);
    if (betaType=='variable'){
      globalStore.posteriorSampleBetas = map(sample,priors.betas);
      if (debug) {console.log(' betas: '+globalStore.posteriorSampleBetas)};
    } else if (betaType=='fixed'){
      var singleBetaSample = sample(priors.betas);
      globalStore.posteriorSampleBetas = 
        mapN(function(n) {return singleBetaSample}, utilities.length);
      
      if (debug) {console.log(' beta: '+singleBetaSample+ ', betas: '
        +globalStore.posteriorSampleBetas)};
    } 
    
    if (debug) {console.log('inside posterior, sample utilities '+utilities+
      ' actions:'+params.actions)};
    
    //create MDP based on known params and sampled utilities and betas
    var MDP = createMDP({actions:params.actions, transitions:params.transitions, 
                     states:params.states, startStates:params.startStates, 
                     utilities:utilities, betas:globalStore.posteriorSampleBetas});
    var agent = softMaxAgent({MDP:MDP, discount:discount});
    var act = agent.act;

    
    // For each observed state-action pair, factor on likelihood of action,
    //given the remaining length
    map2(
      function(stateAction, length){
        var state = stateAction[0];
        var action = stateAction[1];
        observe(act({state:state,length:length }), action);
      },
      traj, remainingLengths);
    return {utilities:utilities, betas:globalStore.posteriorSampleBetas};
  }});
};

var getTotalUtility = function(options){
  assert.ok(_.has(options, 'traj') &&
           _.has(options, 'getUtility'),
           'getTotalUtility args missing one or more of traj, getUtility') 
  var getUtility = options.getUtility;
  var utilities = map(
      function(stateaction){
        var state = stateaction[0];
        return getUtility(state);
      }, options.traj);
    
    var total = sum(utilities);
    return total;
}

//calculate average utility achieved by human
var humanScore = function(options) {
  assert.ok(_.has(options, 'MDP') &&
           _.has(options, 'length') &&
           _.has(options, 'discount'), 'humanScore args: missing one or more'+
           'of MDP, length, discount');
  
  var debug = false;
  if(debug){console.log('calling humanscore');};

  var getUtility = options.MDP.getUtility;
  var agent = softMaxAgent({MDP:options.MDP, discount:options.discount});
  //calculate average utility achieved by human
  return expectation(Infer( {method:"rejection", model(){
    var traj = makeTrajectory({MDP:options.MDP, length:options.length,
                              agent:agent});
    var utility = getTotalUtility({getUtility:getUtility, traj:traj});

    if (debug){
      console.log('human traj: '+traj + ' utility: '+utility);
    }
    return utility;
   }}));
};


//function to count which is the most frequent action for each state
var getCounts = function(options) {
  assert.ok(_.has(options, 'traj') &&
           _.has(options, 'states') &&
           _.has(options, 'getActions'), 'getCounts args: missing one or more '+
           'of traj, states, getActions');
  
  var getActions = options.getActions;
  
   var count = function(state, action){
      var test = function(stateActionPair) {
        return ((state==stateActionPair[0])&&(action==stateActionPair[1])) ? 1 : 0
      }
      var flags = map(test, options.traj); //array containing a 1 for each match
      return sum(flags);
   }
   
   //for each state, list for each available action the counts for that action
   var counts = 
       map(
         function(state){
           return map(
             function(action){
               return count(state, action);
             }, 
             getActions(state)
           )},
         options.states
       );
  
    //console.log(counts);
    return counts;
};
 

var naiveRobotScore = function(options){
  //Naive robot chooses the most frequent action taken in each state
  //in the observed traj
  var debug1=false;
  var debug2=false;
  
  assert.ok(_.has(options, 'MDP') &&
           _.has(options, 'length') &&
           _.has(options, 'discount'), 'naiveRobot Score args: missing one '+
           'or more of MDP, length, discount');
  
  var MDP=options.MDP;
  var getUtility = MDP.getUtility;
  var getActions = MDP.getActions;
  var states = MDP.states;
  var agent = softMaxAgent({MDP:MDP, discount:options.discount});
  var length = options.length;
  
  var robotAction = function(state, counts){
    var index = states.indexOf(state);
    assert.ok((index!=-1), 'robotAction: state not found');
    var actionCounts = counts[index];
    var maxActionCount = reduce(max, -9999999, actionCounts);
    
    //if multiple actions are most frequent, randomly return first or last one
    var mostFrequentAction1 = actionCounts.indexOf(maxActionCount);
    var mostFrequentAction2 = actionCounts.lastIndexOf(maxActionCount);
    var actionIndex = uniformDraw([mostFrequentAction1, mostFrequentAction2]);
    var actions = getActions(state);
    return actions[actionIndex];
  }
  
  console.log('calling naiveRobotScore');
  return expectation(Infer({ model() {  
    var observedTrajectory = makeTrajectory({MDP:MDP, length:length, agent:agent}); 
    var counts = getCounts({traj:observedTrajectory, states:states,
                            getActions:getActions});
    
    
   
    var start = uniformDraw(MDP.startStates);
    
    var step = function(options){
      assert.ok(_.has(options, 'state') &&
                _.has(options, 'length'), 
                'makeTrajectory.step args:' +
                'one or more of length, state is missing');
      var state=options.state;
      var length=options.length;
      var action = robotAction(state, counts);
      var transDist = transition({state:state, action:action})
      var next = sample(transDist);
      var res = [state, action];
      var newLength = length-1;
      
      if (debug1) {console.log('action: ' + action + ' transDist: '+transDist+
                              ' length: '+length +' newLen: '+newLength+
                              ' next: '+ next + ' counts: '+counts);};
      return length==1 ? 
        [res] : 
      [res].concat(step({state:next, length: newLength}));
    };
  
   var robotTraj = step({state:start, length:length}); 
    
   if (debug2) {console.log('obs, counts, robot ' +
                           observedTrajectory, counts, robotTraj);;}
   return getTotalUtility({traj:robotTraj, getUtility:getUtility});
  }}));
};


var IRLRobotScore = function(options){
   assert.ok(_.has(options, 'MDP') &&
             _.has(options, 'length') &&
             _.has(options, 'priors') &&
             _.has(options, 'beta') &&
             _.has(options, 'nSamples') &&
             _.has(options, 'discount'), 'IRLRobotScore args: missing one '+
             'or more of MDP, beta, length, priors, nSamples, discount');
  
  var betaType=options.beta;
  //assert.ok((betaType=='fixed')||(betaType=='variable'), 'beta must be "fixed"+
  //          ' or "variable"');
  
  var MDP=options.MDP;
  var human = softMaxAgent({MDP:MDP, discount:options.discount});
  var length = options.length;
  var MDPParams = ({actions:MDP.actions, transitions:MDP.transitions, 
                     states:MDP.states, startStates:MDP.startStates});
  var getUtility = MDP.getUtility;
  var robotBetas = mapN(function(n){return 0.001}, MDP.states.length); //robot has very low beta
  var debug=false;
  
  return expectation(Infer({method:"rejection", samples:options.nSamples, model(){
    var observedTrajectory = makeTrajectory({MDP:MDP, length:length, agent:human}); 
    var sample = sample(posterior({priors:options.priors, 
                                   discount:options.discount,
                                   observedTrajectory:observedTrajectory,
                                   MDPParams:MDPParams,
                                   beta:betaType}));
    
   
    //robot MDP has sampled utilities and very low betas everywhere
    var robotMDP = createMDP({actions:MDP.actions, transitions:MDP.transitions, 
                     states:MDP.states, startStates:MDP.startStates, 
                     utilities:sample.utilities, betas:robotBetas});
    var robot = softMaxAgent({MDP:robotMDP, discount:options.discount});
    var robotTraj = makeTrajectory({length:length, MDP:robotMDP, 
                               agent:robot});

    if (debug) {
      console.log('traj: '+ observedTrajectory + ' sample utils: ' + sample.utilities+
        ' sampled betas: '+sample.betas+' robot traj: '+robotTraj);
    }
    return getTotalUtility({traj:robotTraj, getUtility:getUtility});

  }}));
};


var simulate2StateMDP = function(len,nSamples){
  //simple 2-state determinstic MDP    <A-B> 

  console.log('traj length: '+len+' nSamples: '+nSamples);

  //define priors
  var utilityPriors = mapN(
    function(n){return Categorical({ps:[1,1], vs:[1,10]})}
    , 2);
  var betaPriors = mapN(
    function(n){Categorical({ps:[1,1], vs:[1,10]})}, 
    2);
  var singleBetaPrior = Categorical({ps:[1,1], vs:[1,10]});
  var variablePriors = {utilities:utilityPriors, betas:betaPriors};
  var fixedPriors = {utilities:utilityPriors, betas:singleBetaPrior};


  console.log('utilityPriors '+utilityPriors+' betaPriors '+
    betaPriors+' singleBetaPrior '+singleBetaPrior);
  //define action set, states, transitions
  var actions = [['A', 'B'], ['A', 'B']];
  var states = ['A', 'B'];
  var startStates = ['A', 'B'];

  var transition = function(options) {
    assert.ok( _.has(options, 'state') && 
               _.has(options, 'action'), 'transition args: missing one or more ' +
                 'of state, action');
    var action=options.action;
    var state=options.state;
    var nextProbs = (action === 'A') ? [1,0] : [0,1];
    return Categorical({ps:nextProbs, vs:states});
  };

  var debug =false;
  
  //Infer over possible MDP setups, drawn from priors
  var humanScoreRes = expectation(Infer({method:"enumerate", model(){
    //sample betas and utilities
    var betas = map(sample,betaPriors);
    var utilities = map(sample,utilityPriors);
    
    var MDP = createMDP({states:states,startStates:startStates, actions:actions, 
                        transitions:transition, utilities:utilities, betas:betas});


    var humanScoreSample = humanScore({length:len, MDP:MDP, discount:0.99});
    
    if(debug){
      console.log('humanScoreSample: '+humanScoreSample+' betas '+betas+' utilities '+utilities);
    };

    return humanScoreSample
  }}));

  var robotScoreFixed = expectation(Infer({method:"enumerate", model(){
    //sample betas and utilities
    var betas = map(sample,betaPriors);
    var utilities = map(sample,utilityPriors);
    
    var MDP = createMDP({states:states,startStates:startStates, actions:actions, 
                        transitions:transition, utilities:utilities, betas:betas});

    var paramsFixed = {MDP:MDP, discount:0.99, length:len, priors:fixedPriors, 
      beta:"fixed", nSamples:nSamples};

    var score = IRLRobotScore(paramsFixed);
    return score
  }}));
    
  var robotScoreVariable = expectation(Infer({method:"enumerate", model(){
    //sample betas and utilities
    var betas = map(sample,betaPriors);
    var utilities = map(sample,utilityPriors);
    
    var MDP = createMDP({states:states,startStates:startStates, actions:actions, 
                        transitions:transition, utilities:utilities, betas:betas});

    var paramsVariable = {MDP:MDP, discount:0.99, length:len, priors:variablePriors,
     beta:"variable", nSamples:nSamples};

    var score = IRLRobotScore(paramsVariable);
    return score
  }}));

  console.log('human '+ humanScoreRes);
  console.log('fixed '+robotScoreFixed);
  console.log('variable '+robotScoreVariable);
}


var simulateSpecific2StateMDP = function(len,nSamples){
  //simple 2-state determinstic MDP    <A-B> 
  

  //define priors
  var utilityPriors = mapN(
    function(n){return Categorical({ps:[1,1], vs:[1,10]})}
    , 2);
  var betaPriors = mapN(
    function(n){Categorical({ps:[1,1], vs:[1,10]})}, 
    2);
  var singleBetaPrior = Categorical({ps:[1,1], vs:[1,10]});
  var variablePriors = {utilities:utilityPriors, betas:betaPriors};
  var fixedPriors = {utilities:utilityPriors, betas:singleBetaPrior};


  //define action set, states, transitions, utilities, betas
  var actions = [['A', 'B'], ['A', 'B']];
  var states = ['A', 'B'];
  var startStates = ['A', 'B'];
  var betas = [1,10];
  var utilities = [1,10];

  var transition = function(options) {
    assert.ok( _.has(options, 'state') && 
               _.has(options, 'action'), 'transition args: missing one or more ' +
                 'of state, action');
    var action=options.action;
    var state=options.state;
    var nextProbs = (action === 'A') ? [1,0] : [0,1];
    return Categorical({ps:nextProbs, vs:states});
  };

  console.log('traj length: '+len+' nSamples: '+nSamples+
    " betas: "+betas+" utilities: "+utilities);

  var MDP = createMDP({states:states,startStates:startStates, actions:actions, 
                        transitions:transition, utilities:utilities, betas:betas}); 
  console.log('created MDP, traj length: '+len);
  
  var paramsFixed = {MDP:MDP, discount:0.99, length:len, priors:fixedPriors, 
      beta:"fixed", nSamples:nSamples};
  console.log('created paramsFixed, traj length: '+len);
  
  var paramsVariable = {MDP:MDP, discount:0.99, length:len, priors:variablePriors,
     beta:"variable", nSamples:nSamples};
  console.log('created paramsVariable, traj length: '+len);
  
  var agent = softMaxAgent({MDP:MDP, discount:0.99});
  console.log('created agent, traj length: '+len);

  var trajs = mapN(function(n){ 
    //console.log('creating example traj, traj length: '+len);
    var trajParams = {length:len, MDP:MDP, agent:agent};
    //console.log(trajParams);
    var exampleTraj= makeTrajectory(trajParams);
    //console.log('created example traj, traj length: '+len);
    return exampleTraj;
  },20);
  
  console.log('created trajs, traj length: '+len);

  console.log('example trajs: ')
  console.log(trajs);

  var humanScoreRes = humanScore({length:len, MDP:MDP, discount:0.99});
  console.log('human '+ humanScoreRes);

  var robotScoreFixed = IRLRobotScore(paramsFixed);
  console.log('fixed '+robotScoreFixed);
  
  var robotScoreVariable = IRLRobotScore(paramsVariable);
  console.log('variable '+robotScoreVariable);

  console.log('finished computing specific MDP with '+
    'traj length: '+len+' nSamples: '+nSamples+
    " betas: "+betas+" utilities: "+utilities);
}

var simulateDangerMDP = function(len,nSamples){
  //simple 2-state determinstic MDP    <A-B> 

  //define priors
  var utilityPriors = mapN(
    function(n){return Categorical({ps:[1,1,1], vs:[1,2,-5]})}
    , 3);
  var betaPriors = mapN(
    function(n){Categorical({ps:[1,1], vs:[1,20]})}, 
    3);
  var singleBetaPrior = Categorical({ps:[1,1], vs:[1,20]});
  var variablePriors = {utilities:utilityPriors, betas:betaPriors};
  var fixedPriors = {utilities:utilityPriors, betas:singleBetaPrior};


  //define action set, states, transitions, utilities, betas
  var actions = [['safe', 'risky', 'bad'], ['safe', 'risky', 'bad'],['safe', 'risky', 'bad']];
  var states = ['safe', 'risky', 'bad'];
  var startStates = ['safe'];
  var betas = [1,20,20];
  var utilities = [1,2,-5];

  var transition = function(options) {
    assert.ok( _.has(options, 'state') && 
               _.has(options, 'action'), 'transition args: missing one or more ' +
                 'of state, action');
    var action=options.action;
    var state=options.state;
    var nextProbs = (action === 'safe') ? [1,0,0] : 
                    (action === 'risky') ?[0,1,0] : [0,0,1];
    return Categorical({ps:nextProbs, vs:states});
  };


  var MDP = createMDP({states:states,startStates:startStates, actions:actions, 
                        transitions:transition, utilities:utilities, betas:betas});


  var paramsFixed = {MDP:MDP, discount:0.99, length:len, priors:fixedPriors, 
      beta:"fixed", nSamples:nSamples};
  var paramsVariable = {MDP:MDP, discount:0.99, length:len, priors:variablePriors,
     beta:"variable", nSamples:nSamples};

  var agent = softMaxAgent({MDP:MDP, discount:0.99});

  var trajs = mapN(function(n){return makeTrajectory({length:len, MDP:MDP, agent:agent})},20);
  console.log('traj length: '+len+' nSamples: '+nSamples+
    " betas: "+betas+" utilities: "+utilities);

  console.log('example trajs: ')
  console.log(trajs);


  var humanScoreRes = humanScore({length:len, MDP:MDP, discount:0.99});
  console.log('human '+ humanScoreRes);

  var robotScoreFixed = IRLRobotScore(paramsFixed);
  console.log('fixed '+robotScoreFixed);

  var robotScoreVariable = IRLRobotScore(paramsVariable);
  console.log('variable '+robotScoreVariable);
}

var trajLengths = [5,8,10,15,20,25];
var nSamplesList = [100, 200];


if(false){
  var sim1 = map(
    function(nSamplesLambda){
      return map(
        function(trajLengthLambda){
          console.log('calling specific with nSamples, '+
            nSamplesLambda+' trajLength '+trajLengthLambda);
          return simulateSpecific2StateMDP(trajLengthLambda, nSamplesLambda);
        }, trajLengths
      );
    }, nSamplesList
    );

  console.log('break');

  var sim2 = map(
    function(nSamples){
      return map(
        function(trajLength){
          return simulate2StateMDP(trajLength, nSamples);
        }, trajLengths
      );
    }, nSamplesList
    );


  console.log('break');


  var sim3= map(
    function(nSamples){
      return map(
        function(trajLength){
          return simulateDangerMDP(trajLength, nSamples);
        }, trajLengths
      );
    }, nSamplesList
    );
}













//define priors
  var utilityPriors = mapN(
    function(n){return Categorical({ps:[1,1,1], vs:[1,2,-5]})}
    , 3);
  var betaPriors = mapN(
    function(n){Categorical({ps:[1,1], vs:[1,20]})}, 
    3);
  var singleBetaPrior = Categorical({ps:[1,1], vs:[1,20]});
  var variablePriors = {utilities:utilityPriors, betas:betaPriors};
  var fixedPriors = {utilities:utilityPriors, betas:singleBetaPrior};


  //define action set, states, transitions, utilities, betas
  var actions = [['safe', 'risky', 'bad'], ['safe', 'risky', 'bad'],['safe', 'risky', 'bad']];
  var states = ['safe', 'risky', 'bad'];
  var startStates = ['safe'];
  var betas = [0.1,20,20];
  var utilities = [1,2,-5];

  var transition = function(options) {
    assert.ok( _.has(options, 'state') && 
               _.has(options, 'action'), 'transition args: missing one or more ' +
                 'of state, action');
    var action=options.action;
    var state=options.state;
    var nextProbs = (action === 'safe') ? [1,0,0] : 
                    (action === 'risky') ?[0,1,0] : [0,0,1];
    return Categorical({ps:nextProbs, vs:states});
  };


  var MDP = createMDP({states:states,startStates:startStates, actions:actions, 
                        transitions:transition, utilities:utilities, betas:betas});

  var agent = softMaxAgent({MDP:MDP, discount:0.99});

  
  var trajs = mapN(function(n){
    return makeTrajectory({length:5, MDP:MDP, agent:agent})},20);

  console.log(trajs)
  console.log(agent);




