# this is only the matching, then i want to include other functions to 
# check that the matches are the most stable ones (i will be refering to existing
# code and pseudocode in order to keep the cleanest implementation)

def stable_matching(men, women, men_prefs, women_prefs):
    # Initialize everyone as free
    free_men = set(men)
    engagements = {}
    # Store the count of proposals to avoid modifying original lists
    proposals_count = {man: 0 for man in men}

    # Continue while there is at least one free man who can still propose
    while free_men:
        for man in list(free_men):  # Iterate over a snapshot of free men
            # Get the list of preferences for the man
            pref_list = men_prefs[man]
            # Find the next woman to propose to
            if proposals_count[man] < len(pref_list):
                woman = pref_list[proposals_count[man]]
                proposals_count[man] += 1
            else:
                continue  # No more women to propose to

            # Check if the woman is free or engaged
            if woman not in engagements:
                # If the woman is free, engage her with the man
                engagements[woman] = man
                free_men.remove(man)
                print(f"{man} and {woman} are now engaged.")
            else:
                # If the woman is already engaged
                current_man = engagements[woman]
                # Determine if the woman prefers the new man over her current engagement
                if women_prefs[woman].index(man) < women_prefs[woman].index(current_man):
                    # The woman prefers the new man, change the engagement
                    print(f"{woman} dumped {current_man} for {man}.")
                    engagements[woman] = man
                    free_men.add(current_man)  # Current man is now free
                    free_men.remove(man)       # New man is now engaged
                # If the woman prefers her current engagement, do nothing
                # Current man stays engaged, new man remains free and continues proposing

    return engagements


def get_preferred_partners(preferences, current_partner):
    index = preferences.index(current_partner)
    return preferences[:index]


def check(engaged):
    inverseengaged = dict((v,k) for k,v in engaged.items())
    for she, he in engaged.items():
        shelikes = galprefers[she]
        shelikesbetter = shelikes[:shelikes.index(he)]
        helikes = guyprefers[he]
        helikesbetter = helikes[:helikes.index(she)]
        for guy in shelikesbetter:
            guysgirl = inverseengaged[guy]
            guylikes = guyprefers[guy]
            if guylikes.index(guysgirl) > guylikes.index(she):
                print("%s and %s like each other better than "
                      "their present partners: %s and %s, respectively"
                      % (she, guy, he, guysgirl))
                return False
        for gal in helikesbetter:
            girlsguy = engaged[gal]
            gallikes = galprefers[gal]
            if gallikes.index(girlsguy) > gallikes.index(he):
                print("%s and %s like each other better than "
                      "their present partners: %s and %s, respectively"
                      % (he, gal, she, girlsguy))
                return False
    return True


# def check_engagement(engagements):
#     received_engagement =  {v: k, for k, v in engagements.items()}
#     for proposer, receiver in engagements.items():
#         preferred_proposers = get_preferred_partners(
#             receiver_prefers[receiver], proposer)
#         preferred_receivers = get_preferred_partners(
#             proposer_prefers[proposer], receiver)
#         for prop in preferred_proposers:
#             main_proposer = received_engagement[prop]
#              = 
            


# def is_stable_pair(engagements, receiver_prefers, proposer_prefers):
#     inverse_engaged = {v: k for k, v in engagements.items()}
#     for receiver, proposer in engagements.items():
#         preferred_proposers = get_preferred_partners(
#             receiver_prefers[receiver], proposer)
#         preferred_receivers = get_preferred_partners(
#             proposer_prefers[proposer], receiver)

#         if any_prefers_current(
#                 engagements, 
#                 receiver, 
#                 proposer, 
#                 preferred_proposers, 
#                 preferred_receivers, 
#                 inverse_engaged, 
#                 receiver_prefers, 
#                 proposer_prefers
#                 ):
#             return False

#     return True

# def any_prefers_current(engagements, receiver, proposer, preferred_proposers, preferred_receivers, inverse_engaged, receiver_prefers, proposer_prefers):
#     for other_prop in preferred_proposers:
#         other_proposer_partner = inverse_engaged[other_prop]
#         if proposer_prefers[other_prop].index(other_proposer_partner) > proposer_prefers[other_prop].index(receiver):
#             print(f"{receiver} and {other_prop} like each other better than their present partners: {proposer} and {other_proposer_partner}, respectively")
#             return True

#     for other_rec in preferred_receivers:
#         other_receiver_partner = engagements[other_rec]
#         if receiver_prefers[other_rec].index(other_receiver_partner) > receiver_prefers[other_rec].index(proposer):
#             print(f"{proposer} and {other_rec} like each other better than their present partners: {receiver} and {other_receiver_partner}, respectively")
#             return True

#     return False

engagements1 = stable_matching(a, b, pref_reg_a, pref_reg_b)
    
def is_stable_pair(engaged, galprefers, guyprefers):
    inverse_engaged = {v: k for k, v in engaged.items()}

    for woman, man in engaged.items():
        preferred_men = galprefers[woman][:galprefers[woman].index(man)]
        preferred_women = guyprefers[man][:guyprefers[man].index(woman)]

        if any_prefers_current(engaged, woman, man, preferred_men, preferred_women, inverse_engaged, galprefers, guyprefers):
            return False

    return True

def any_prefers_current(engaged, woman, man, preferred_men, preferred_women, inverse_engaged, galprefers, guyprefers):
    for other_man in preferred_men:
        other_man_partner = inverse_engaged[other_man]
        if guyprefers[other_man].index(other_man_partner) > guyprefers[other_man].index(woman):
            print(f"{woman} and {other_man} like each other better than their present partners: {man} and {other_man_partner}, respectively")
            return True

    for other_woman in preferred_women:
        other_woman_partner = engaged[other_woman]
        if galprefers[other_woman].index(other_woman_partner) > galprefers[other_woman].index(man):
            print(f"{man} and {other_woman} like each other better than their present partners: {woman} and {other_woman_partner}, respectively")
            return True

    return False

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Check if the current engagements are stable
stability = is_stable_pair(engagements1, pref_reg_b, pref_reg_a)
print("Is the engagement stable?", stability)


pref_reg_a = {1: [19, 20, 12, 11, 17, 18, 15, 16, 13, 14], 2: [17, 15, 20, 19, 14, 12, 13, 18, 16, 11], 3: [17, 20, 14, 11, 12, 13, 15, 16, 18, 19], 4: [19, 14, 20, 12, 15, 17, 18, 13, 11, 16], 5: [14, 20, 19, 15, 12, 13, 18, 16, 11, 17], 6: [14, 20, 19, 11, 17, 12, 15, 18, 13, 16], 7: [20, 17, 19, 14, 13, 16, 18, 11, 12, 15], 8: [20, 12, 14, 13, 17, 16, 11, 18, 15, 19], 9: [12, 11, 20, 16, 14, 19, 17, 18, 15, 13], 10: [20, 19, 11, 12, 13, 14, 15, 16, 17, 18]}

pref_reg_b = {11: [2, 1, 3, 4, 5, 6, 7, 8, 9, 10], 12: [9, 7, 8, 2, 4, 6, 3, 5, 1, 10], 13: [9, 2, 4, 3, 8, 6, 7, 1, 5, 10], 14: [2, 4, 8, 3, 1, 9, 5, 6, 7, 10], 15: [2, 4, 7, 1, 6, 5, 3, 8, 9, 10], 16: [8, 7, 1, 4, 9, 5, 6, 2, 3, 10], 17: [9, 4, 7, 8, 3, 6, 1, 2, 5, 10], 18: [8, 7, 9, 2, 4, 5, 1, 3, 6, 10], 19: [7, 8, 2, 1, 3, 10, 9, 6, 5, 4], 20: [8, 7, 6, 2, 1, 9, 3, 10, 5, 4]}












guyprefers = {
 'abe':  ['abi', 'eve', 'cath', 'ivy', 'jan', 'dee', 'fay', 'bea', 'hope', 'gay'],
 'bob':  ['cath', 'hope', 'abi', 'dee', 'eve', 'fay', 'bea', 'jan', 'ivy', 'gay'],
 'col':  ['hope', 'eve', 'abi', 'dee', 'bea', 'fay', 'ivy', 'gay', 'cath', 'jan'],
 'dan':  ['ivy', 'fay', 'dee', 'gay', 'hope', 'eve', 'jan', 'bea', 'cath', 'abi'],
 'ed':   ['jan', 'dee', 'bea', 'cath', 'fay', 'eve', 'abi', 'ivy', 'hope', 'gay'],
 'fred': ['bea', 'abi', 'dee', 'gay', 'eve', 'ivy', 'cath', 'jan', 'hope', 'fay'],
 'gav':  ['gay', 'eve', 'ivy', 'bea', 'cath', 'abi', 'dee', 'hope', 'jan', 'fay'],
 'hal':  ['abi', 'eve', 'hope', 'fay', 'ivy', 'cath', 'jan', 'bea', 'gay', 'dee'],
 'ian':  ['hope', 'cath', 'dee', 'gay', 'bea', 'abi', 'fay', 'ivy', 'jan', 'eve'],
 'jon':  ['abi', 'fay', 'jan', 'gay', 'eve', 'bea', 'dee', 'cath', 'ivy', 'hope']}
galprefers = {
 'abi':  ['bob', 'fred', 'jon', 'gav', 'ian', 'abe', 'dan', 'ed', 'col', 'hal'],
 'bea':  ['bob', 'abe', 'col', 'fred', 'gav', 'dan', 'ian', 'ed', 'jon', 'hal'],
 'cath': ['fred', 'bob', 'ed', 'gav', 'hal', 'col', 'ian', 'abe', 'dan', 'jon'],
 'dee':  ['fred', 'jon', 'col', 'abe', 'ian', 'hal', 'gav', 'dan', 'bob', 'ed'],
 'eve':  ['jon', 'hal', 'fred', 'dan', 'abe', 'gav', 'col', 'ed', 'ian', 'bob'],
 'fay':  ['bob', 'abe', 'ed', 'ian', 'jon', 'dan', 'fred', 'gav', 'col', 'hal'],
 'gay':  ['jon', 'gav', 'hal', 'fred', 'bob', 'abe', 'col', 'ed', 'dan', 'ian'],
 'hope': ['gav', 'jon', 'bob', 'abe', 'ian', 'dan', 'hal', 'ed', 'col', 'fred'],
 'ivy':  ['ian', 'col', 'hal', 'gav', 'fred', 'bob', 'abe', 'ed', 'jon', 'dan'],
 'jan':  ['ed', 'hal', 'gav', 'abe', 'bob', 'jon', 'col', 'ian', 'fred', 'dan']}

guys = sorted(guyprefers.keys())
gals = sorted(galprefers.keys())


a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

pref_a = {
    1: [20, 13, 12, 19, 14, 17, 18, 15, 16, 11],
    2: [14, 20, 12, 19, 15, 17, 13, 16, 18, 11],
    3: [14, 19, 20, 12, 17, 15, 13, 18, 11, 16],
    4: [14, 20, 12, 19, 17, 13, 15, 11, 16, 18],
    5: [14, 20, 19, 12, 15, 17, 13, 18, 11, 16],
    6: [20, 14, 12, 15, 13, 19, 17, 18, 16, 11],
    7: [14, 20, 12, 19, 15, 17, 13, 16, 18, 11],
    8: [14, 12, 19, 15, 20, 17, 13, 18, 11, 16],
    9: [14, 20, 12, 19, 15, 17, 13, 16, 18, 11],
    10: [20, 14, 12, 19, 15, 13, 17, 11, 18, 16]}
pref_b =  {
    11: [2, 8, 7, 9, 6, 4, 3, 5, 1, 10],
    12: [8, 7, 9, 4, 2, 6, 5, 3, 1, 10],
    13: [8, 9, 4, 10, 7, 5, 2, 3, 6, 1],
    14: [2, 9, 4, 8, 6, 7, 10, 3, 5, 1],
    15: [8, 7, 2, 6, 3, 9, 4, 5, 1, 10],
    16: [8, 2, 9, 7, 4, 6, 3, 10, 1, 5],
    17: [2, 7, 8, 9, 6, 4, 3, 5, 1, 10],
    18: [8, 9, 7, 6, 2, 4, 10, 3, 5, 1],
    19: [8, 7, 2, 9, 4, 3, 6, 10, 1, 5],
    20: [8, 9, 7, 2, 3, 6, 4, 10, 5, 1]}




pref_c = {
    1: [19, 12, 20, 17, 11, 15, 18, 16, 13, 14], 
    2: [17, 19, 20, 15, 14, 12, 13, 16, 18, 11], 
    3: [17, 14, 11, 12, 20, 13, 15, 16, 18, 19], 
    4: [19, 14, 20, 12, 15, 13, 17, 18, 11, 16], 
    5: [14, 19, 20, 13, 15, 12, 18, 16, 17, 11], 
    6: [20, 14, 19, 17, 11, 12, 18, 15, 13, 16], 
    7: [17, 20, 14, 13, 19, 11, 16, 18, 15, 12], 
    8: [12, 20, 14, 13, 17, 11, 16, 18, 15, 19], 
    9: [12, 11, 16, 20, 14, 17, 18, 19, 15, 13], 
    10: [20, 19, 11, 12, 13, 14, 15, 16, 17, 18]}

pref_d = {
    11: [2, 1, 3, 4, 5, 6, 7, 8, 9, 10], 
    12: [9, 7, 8, 4, 2, 3, 6, 5, 1, 10], 
    13: [3, 9, 2, 4, 8, 7, 6, 1, 5, 10], 
    14: [2, 4, 8, 3, 1, 9, 5, 6, 7, 10], 
    15: [2, 4, 7, 1, 6, 5, 3, 8, 9, 10], 
    16: [7, 8, 9, 4, 1, 2, 6, 5, 3, 10], 
    17: [9, 4, 7, 6, 8, 1, 3, 2, 5, 10], 
    18: [8, 7, 2, 9, 4, 5, 1, 3, 6, 10], 
    19: [7, 8, 2, 1, 3, 10, 9, 6, 5, 4], 
    20: [8, 7, 6, 2, 1, 3, 10, 9, 5, 4]}












