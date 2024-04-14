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