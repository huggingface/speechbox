.PHONY: deps_table_update modified_only_fixup extra_style_checks quality style fixup

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := examples src

# Update src/diffusers/dependency_versions_table.py

deps_table_update:
	@python setup.py deps_table_update

deps_table_check_updated:
	@md5sum src/diffusers/dependency_versions_table.py > md5sum.saved
	@python setup.py deps_table_update
	@md5sum -c --quiet md5sum.saved || (printf "\nError: the version dependency table is outdated.\nPlease run 'make fixup' or 'make style' and commit the changes.\n\n" && exit 1)
	@rm md5sum.saved

# autogenerating code

autogenerate_code: deps_table_update

# Check that the repo is in a good state

# this target runs checks on all files

quality:
	black --check --preview $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

# Format source code automatically and check is there are any problems left that need manual fixing

# this target runs checks on all files and potentially modifies some of them

style:
	black --preview $(check_dirs)
	isort $(check_dirs)
	${MAKE} autogenerate_code

fix-copies:
	python utils/check_dummies.py --fix_and_overwrite
