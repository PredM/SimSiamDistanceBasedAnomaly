features_all_cases = data['relevant_features']
all_cases = data['all_cases']

# Reduce features of all cases to the subset of cases configured in self.cases_used
if self.cases_used is None or len(self.cases_used) == 0:
    self.cases_used = all_cases
    self.relevant_features = features_all_cases
else:
    self.relevant_features = {case: features_all_cases[case] for case in self.cases_used if
                              case in features_all_cases}

# sort feature names to ensure that the order matches the one in the list of indices of the features in
# the case base class
for key in self.relevant_features:
    self.relevant_features[key] = sorted(self.relevant_features[key])