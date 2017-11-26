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
    assert.ok((index!=-1), 'state not found');
    return options.actions[index];
  };
  
  var getUtility = function(state){
    var index = options.states.indexOf(state);
    assert.ok((index!=-1), 'state not found');
    return options.utilities[index];
  };
  
  var getTransition = options.transitions;
  var startStates = options.startStates;
  var length = options.length;
  
  var getExpectedUtility = function(options) {
       assert.ok(_.has(options, 'state') &&
            _.has(options, 'action'), 
                 'getExpectedUtility args:' +
                 'one or more of state, action is missing');
    var expected =  expectation(Infer({model() { 
         var current = options.state;
         var action = options.action;
         var next = sample(getTransition({state:current, action:action}));
         var util = getUtility(next);
         return util;

       }}));     
    return expected;
    

  };
  
  return{getBeta, getActions, getUtility, getExpectedUtility,
         getTransition, startStates, states};
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
  
  var getEU = function(options) {
    //console.log('called eu');
       assert.ok(_.has(options, 'state') &&
                 _.has(options, 'action') &&
                 _.has(options, 'length'), 
                 'getExpectedUtility args:' +
                 'one or more of state, action is missing');
    var expected =  expectation(Infer({method:'enumerate', model() { 
       var current = options.state;
       var action = options.action;
       var next = sample(getTransition({state:current, action:action}));
       var util = getUtility(next);
       if (length==0){
         return util;
       } else{
         var nextAction = act({state:next, length:length});
         var newLength = length-1;
         var futureUtil = discount*getEU({
           state:next, action:nextAction, length:newLength})
         return (util + futureUtil);

       }}}));  
    return expected;
  };
  
  var act = function(options){
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
            console.log("action: " + action + " state: " + state + 
                 " beta: " + beta + " eu: "+eu+" factor: "+eu/beta );
          }
          factor(eu/beta);
          
        return action;
        }
   });
  };
  return {getEU, act};
};

//generate sample trajectory
var makeTrajectory = function(options) {
  //console.log('calling maketraj');
   assert.ok(_.has(options, 'length') &&
             _.has(options, 'agent') &&
            _.has(options, 'MDP'), 
                 'makeTrajectory args:' +
                 'one or more of length, agent, MDP is missing');
  var MDP = options.MDP;
  var length = options.length;
  var start = uniformDraw(MDP.startStates);
  //console.log('start is ' + start + ' length is '+ length);
  var agent = options.agent;
  var act=agent.act;
  var transition=MDP.getTransition;
  
  var step = function(options){
    //console.log('calling step');
    assert.ok(_.has(options, 'state') &&
              _.has(options, 'length'), 
                 'makeTrajectory.step args:' +
                 'one or more of length, state is missing');
    var state=options.state;
    var length=options.length;
    var action = sample(act({state:state, length:length}));
    var transDist = transition({state:state, action:action})
    var next = sample(transDist);
    var res = [state, action];
    var newLength = length-1;
    var debug=false;
    if (debug) {console.log('action: ' + action + ' trnsDist: '+transDist+
               ' length: '+length +' newLen: '+newLength+ ' next: '+ next);};
    return length==1 ? 
    [res] : 
    [res].concat(step({state:next, length: newLength}));
  };
  
  return step({state:start, length:length});
  
};


var posterior = function(options){
  assert.ok(_.has(options, 'priors') &&
            _.has(options, 'MDPParams') &&
            _.has(options, 'discount') &&
           _.has(options, 'observedTrajectory'),
           'posterior args missing one or more of ' +
            'priors, MDPParams,discount, observedTrajectory');
  
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
    var betas = map(sample,priors.betas);
    
    if (debug) {console.log('sample utilities '+utilities+', betas: '+betas)}
    
    //create MDP based on known params and sampled utilities and betas
    var MDP = createMDP({actions:params.actions, transitions:params.transitions, 
                     states:params.states, startStates:params.startStates, 
                     utilities:utilities, betas:betas});
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
    return {utilities:utilities, betas:betas};
  }});
};

//calculate average utility achieved by human
var humanScore = function(options) {
  assert.ok(_.has(options, 'MDP') &&
           _.has(options, 'length') &&
           _.has(options, 'discount'), 'humanScore args: missing one or more'+
           'of MDP, length, discount');
  
  var getUtility = options.MDP.getUtility;
  var agent = softMaxAgent({MDP:options.MDP, discount:options.discount});
  
  //calculate average utility achieved by human
  return expectation(Infer( { model(){
    var traj = makeTrajectory({MDP:options.MDP, length:options.length,
                              agent:agent});
    var utilities = map(
      function(stateaction){
        var state = stateaction[0];
        return getUtility(state);
      }, traj);
    
    var total = sum(utilities);
    return total;
   }}));
};

var actions = [['A', 'B'],['A', 'B'], ['A', 'B'], ['A', 'B']];
var transition = function(options) {
  assert.ok( _.has(options, 'state') && 
             _.has(options, 'action'), 'transition args: missing one or more ' +
           'of state, action');
  var action=options.action;
  var state=options.state;
  var nextProbs = (action === 'A') ? [0,0,1,0] : [0,0,0,1];
  return Categorical({ps:nextProbs, vs:states});
};

var states = ['start1', 'start2', 'A', 'B'];
var startStates = ['start1','start2'];
var betas = [1,1,1,10];
var utilities = [1,1,1,2];

var params = {actions:actions, transitions:transition, states:states, 
              startStates:startStates};

var utilityPriors = mapN(
  function(n){return Categorical({ps:[2,1], vs:[1,20]})}
  , 4);
var betaPriors = mapN(
  function(n){Categorical({ps:[1,1], vs:[1,10]})}, 
  4);
var priors = {utilities:utilityPriors, betas:betaPriors};

console.log(utilityPriors, betaPriors);

var MDP = createMDP({states:states,startStates:startStates, actions:actions, 
                    transitions:transition, utilities:utilities, betas:betas});

var agent = softMaxAgent({discount:0.9, MDP:MDP});
var traj = makeTrajectory({length:10, agent:agent, MDP:MDP});

print(traj);

//var post = posterior({priors:priors, MDPParams:params, 
//                      discount:0.9, observedTrajectory:traj});
//viz(post);

var score = humanScore({MDP:MDP, discount:0.9, length:9});
print(score);