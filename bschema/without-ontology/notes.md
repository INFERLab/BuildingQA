Need to remove some extra statements from b59 since it was compiled with a different version of 223P

Also needed to remove some specific stuff from brick and un-inference it since it was expanded in multiple ways

Cleaned up the b59 models since there was some shacl and owl left in them for some reason.

in b59 model also removing the extra inferred classes (superclasses to what is present)

May want to make this formally a part of the bschema (or a guideline for summarization) inferred information should probably not be present in the summary