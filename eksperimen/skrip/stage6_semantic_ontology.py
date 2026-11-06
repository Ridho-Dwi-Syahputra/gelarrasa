"""Public model-ontology correspondence, never derived from competition images.

Material names come from the task and MINC. Object names are upstream ImageNet
labels. Object-to-material entries are weak priors, not statements that every
object has that material. The evidence retains disagreement and uncertainty.
"""
import numpy as np
from scipy.special import softmax

CLASSES=['glass','paper','electronic','metal','plastic','cardboard','textile','foodwaste']
OBJECTS={
 'glass':['beer bottle','beaker','goblet','wine bottle','beer glass'],
 'paper':['comic book','book jacket','crossword puzzle','envelope','menu','paper towel','toilet tissue'],
 'electronic':['computer keyboard','cellular telephone','desktop computer','digital clock','electric fan','electric guitar','espresso maker','hair dryer','iPod','iron','laptop','microphone','microwave','modem','monitor','mouse','oscilloscope','photocopier','printer','projector','radio','reflex camera','refrigerator','remote control','sewing machine','space heater','stove','switch','tape player','television','toaster','vacuum','waffle iron','washer','CD player','cassette player','dial telephone','loudspeaker','power drill'],
 'metal':['ashcan','barrel','bottlecap','can opener','chain','cocktail shaker','coffeepot','corkscrew','frying pan','milk can','nail','screw','thimble','wok','padlock','safety pin','screwdriver','wrench','hatchet','hammer','steel drum','soup bowl'],
 'plastic':['water bottle','pop bottle','pill bottle','plastic bag','bucket','soap dispenser','water jug','syringe','rubber eraser','ping-pong ball','tennis ball'],
 'cardboard':['carton'],
 'textile':['abaya','apron','bath towel','bikini','bow tie','brassiere','cardigan','Christmas stocking','diaper','dishrag','doormat','feather boa','fur coat','gown','handkerchief','jersey','kimono','knee pad','lab coat','maillot','mitten','pajama','poncho','quilt','running shoe','sandal','sarong','ski mask','sleeping bag','sock','stole','suit','sweatshirt','swimming trunks','trench coat','velvet','vestment','wool','backpack','purse','pillow','teddy'],
 'foodwaste':['guacamole','consomme','hot pot','trifle','ice cream','ice lolly','French loaf','bagel','pretzel','cheeseburger','hotdog','mashed potato','head cabbage','broccoli','cauliflower','zucchini','spaghetti squash','acorn squash','butternut squash','cucumber','artichoke','bell pepper','cardoon','mushroom','Granny Smith','strawberry','orange','lemon','fig','pineapple','banana','jackfruit','custard apple','pomegranate','hay','carbonara','chocolate sauce','dough','meat loaf','pizza','potpie','burrito','red wine','espresso','eggnog','crab','American lobster','spiny lobster','crayfish','Dungeness crab','rock crab','fiddler crab','sea cucumber']}

MATERIALS={'glass':['glass'],'paper':['paper'],'metal':['metal'],'plastic':['plastic'],
           'textile':['fabric','leather'],'foodwaste':['food']}


def object_evidence(logits,categories):
    probability=softmax(logits,axis=1)
    lookup={name:i for i,name in enumerate(categories)}
    result=np.zeros((len(logits),8),dtype=np.float32)
    missing={}
    for i,name in enumerate(CLASSES):
        available=[lookup[label] for label in OBJECTS[name] if label in lookup]
        missing[name]=[label for label in OBJECTS[name] if label not in lookup]
        assert available,name
        result[:,i]=probability[:,available].max(1)
    return result,missing


def combine_evidence(logits,categories,materials,material_categories):
    objects,missing=object_evidence(logits,categories)
    scores=objects*.35
    index={name:i for i,name in enumerate(material_categories)}
    for label,source in MATERIALS.items():
        scores[:,CLASSES.index(label)]+=.65*materials[:,[index[k] for k in source]].sum(1)
    # MINC has no electronic/cardboard class. Upstream object evidence supplies
    # those categories, rather than reusing a visually named old group.
    for label in ['electronic','cardboard']:
        scores[:,CLASSES.index(label)]=1.5*objects[:,CLASSES.index(label)]
    scores=np.maximum(scores,1e-8)
    return scores/scores.sum(1,keepdims=True),objects,missing


def combine_calibrated(logits,categories,materials,material_categories):
    """Normalize different label spaces before combining evidence.

    Electronic/cardboard are whole-object task categories, so a strong upstream
    object identification takes priority over the casing/constituent material.
    This correspondence is generic ontology logic, not an image-label override.
    """
    objects,missing=object_evidence(logits,categories)
    op=objects/np.maximum(objects.sum(1,keepdims=True),1e-8)
    coverage=np.clip(objects.max(1)/np.maximum(softmax(logits,axis=1).max(1),1e-8),0,1)
    material=np.zeros_like(op)
    index={name:i for i,name in enumerate(material_categories)}
    for name,source in MATERIALS.items():
        material[:,CLASSES.index(name)]=materials[:,[index[k] for k in source]].sum(1)
    mapped=material.sum(1)
    mp=material/np.maximum(mapped[:,None],1e-8)
    # Upstream1000-class probabilities and upstream23-class probabilities are
    # not directly on a comparable scale; use conditional evidence and coverage.
    object_weight=.75*coverage/(.75*coverage+.25*mapped+1e-8)
    score=object_weight[:,None]*op+(1-object_weight[:,None])*mp
    for name in ['electronic','cardboard']:
        i=CLASSES.index(name)
        other=op.copy();other[:,i]=0
        strong=(op[:,i]*coverage>=.35)&((op[:,i]-other.max(1))>=.15)
        score[strong]=.9*op[strong]+.1*mp[strong]
    score=np.maximum(score,1e-8)
    return score/score.sum(1,keepdims=True),objects,missing
