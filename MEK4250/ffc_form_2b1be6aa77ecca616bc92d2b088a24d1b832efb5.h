// This code conforms with the UFC specification version 1.4.0
// and was automatically generated by FFC version 1.4.0.
// 
// This code was generated with the following parameters:
// 
//   cache_dir:                      ''
//   convert_exceptions_to_warnings: False
//   cpp_optimize:                   True
//   cpp_optimize_flags:             '-O2'
//   epsilon:                        1e-14
//   error_control:                  False
//   form_postfix:                   False
//   format:                         'ufc'
//   log_level:                      25
//   log_prefix:                     ''
//   name:                           'ffc'
//   no-evaluate_basis_derivatives:  True
//   optimize:                       False
//   output_dir:                     '.'
//   precision:                      15
//   quadrature_degree:              -1
//   quadrature_rule:                'auto'
//   representation:                 'auto'
//   restrict_keyword:               ''
//   split:                          False

#ifndef __FFC_FORM_2B1BE6AA77ECCA616BC92D2B088A24D1B832EFB5_H
#define __FFC_FORM_2B1BE6AA77ECCA616BC92D2B088A24D1B832EFB5_H

#include <cmath>
#include <stdexcept>
#include <fstream>
#include <ufc.h>

/// This class defines the interface for a finite element.

class ffc_form_2b1be6aa77ecca616bc92d2b088a24d1b832efb5_finite_element_0: public ufc::finite_element
{
public:

  /// Constructor
  ffc_form_2b1be6aa77ecca616bc92d2b088a24d1b832efb5_finite_element_0() : ufc::finite_element()
  {
    // Do nothing
  }

  /// Destructor
  virtual ~ffc_form_2b1be6aa77ecca616bc92d2b088a24d1b832efb5_finite_element_0()
  {
    // Do nothing
  }

  /// Return a string identifying the finite element
  virtual const char* signature() const
  {
    return "FiniteElement('Lagrange', Domain(Cell('interval', 1)), 5, None)";
  }

  /// Return the cell shape
  virtual ufc::shape cell_shape() const
  {
    return ufc::interval;
  }

  /// Return the topological dimension of the cell shape
  virtual std::size_t topological_dimension() const
  {
    return 1;
  }

  /// Return the geometric dimension of the cell shape
  virtual std::size_t geometric_dimension() const
  {
    return 1;
  }

  /// Return the dimension of the finite element function space
  virtual std::size_t space_dimension() const
  {
    return 6;
  }

  /// Return the rank of the value space
  virtual std::size_t value_rank() const
  {
    return 0;
  }

  /// Return the dimension of the value space for axis i
  virtual std::size_t value_dimension(std::size_t i) const
  {
    return 1;
  }

  /// Evaluate basis function i at given point x in cell (actual implementation)
  static void _evaluate_basis(std::size_t i,
                              double* values,
                              const double* x,
                              const double* vertex_coordinates,
                              int cell_orientation)
  {
    // Compute Jacobian
    double J[1];
    compute_jacobian_interval_1d(J, vertex_coordinates);
    
    // Compute Jacobian inverse and determinant
    double K[1];
    double detJ;
    compute_jacobian_inverse_interval_1d(K, detJ, J);
    
    
    // Get coordinates and map to the reference (FIAT) element
    double X = (2.0*x[0] - vertex_coordinates[0] - vertex_coordinates[1]) / J[0];
    
    // Reset values
    *values = 0.0;
    switch (i)
    {
    case 0:
      {
        
      // Array of basisvalues
      double basisvalues[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      
      // Declare helper variables
      
      // Compute basisvalues
      basisvalues[0] = 1.0;
      basisvalues[1] = X;
      basisvalues[2] = X*basisvalues[1]*1.5 - basisvalues[0]*0.5;
      basisvalues[3] = X*basisvalues[2]*1.66666666666667 - basisvalues[1]*0.666666666666667;
      basisvalues[4] = X*basisvalues[3]*1.75 - basisvalues[2]*0.75;
      basisvalues[5] = X*basisvalues[4]*1.8 - basisvalues[3]*0.8;
      for (unsigned int r = 0; r < 6; r++)
      {
        basisvalues[r] *= std::sqrt((0.5 + r));
      } // end loop over 'r'
      
      // Table(s) of coefficients
      static const double coefficients0[6] = \
      {0.0932988114065582, -0.134867738813955, 0.156859010921051, -0.123732056440937, 0.0876868528257127, -0.0440643014954552};
      
      // Compute value(s)
      for (unsigned int r = 0; r < 6; r++)
      {
        *values += coefficients0[r]*basisvalues[r];
      } // end loop over 'r'
        break;
      }
    case 1:
      {
        
      // Array of basisvalues
      double basisvalues[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      
      // Declare helper variables
      
      // Compute basisvalues
      basisvalues[0] = 1.0;
      basisvalues[1] = X;
      basisvalues[2] = X*basisvalues[1]*1.5 - basisvalues[0]*0.5;
      basisvalues[3] = X*basisvalues[2]*1.66666666666667 - basisvalues[1]*0.666666666666667;
      basisvalues[4] = X*basisvalues[3]*1.75 - basisvalues[2]*0.75;
      basisvalues[5] = X*basisvalues[4]*1.8 - basisvalues[3]*0.8;
      for (unsigned int r = 0; r < 6; r++)
      {
        basisvalues[r] *= std::sqrt((0.5 + r));
      } // end loop over 'r'
      
      // Table(s) of coefficients
      static const double coefficients0[6] = \
      {0.0932988114065582, 0.134867738813955, 0.156859010921051, 0.123732056440937, 0.0876868528257128, 0.0440643014954552};
      
      // Compute value(s)
      for (unsigned int r = 0; r < 6; r++)
      {
        *values += coefficients0[r]*basisvalues[r];
      } // end loop over 'r'
        break;
      }
    case 2:
      {
        
      // Array of basisvalues
      double basisvalues[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      
      // Declare helper variables
      
      // Compute basisvalues
      basisvalues[0] = 1.0;
      basisvalues[1] = X;
      basisvalues[2] = X*basisvalues[1]*1.5 - basisvalues[0]*0.5;
      basisvalues[3] = X*basisvalues[2]*1.66666666666667 - basisvalues[1]*0.666666666666667;
      basisvalues[4] = X*basisvalues[3]*1.75 - basisvalues[2]*0.75;
      basisvalues[5] = X*basisvalues[4]*1.8 - basisvalues[3]*0.8;
      for (unsigned int r = 0; r < 6; r++)
      {
        basisvalues[r] *= std::sqrt((0.5 + r));
      } // end loop over 'r'
      
      // Table(s) of coefficients
      static const double coefficients0[6] = \
      {0.368284781867994, -0.516385486449827, 0.188230813105261, 0.0618660282204688, -0.263060558477138, 0.220321507477276};
      
      // Compute value(s)
      for (unsigned int r = 0; r < 6; r++)
      {
        *values += coefficients0[r]*basisvalues[r];
      } // end loop over 'r'
        break;
      }
    case 3:
      {
        
      // Array of basisvalues
      double basisvalues[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      
      // Declare helper variables
      
      // Compute basisvalues
      basisvalues[0] = 1.0;
      basisvalues[1] = X;
      basisvalues[2] = X*basisvalues[1]*1.5 - basisvalues[0]*0.5;
      basisvalues[3] = X*basisvalues[2]*1.66666666666667 - basisvalues[1]*0.666666666666667;
      basisvalues[4] = X*basisvalues[3]*1.75 - basisvalues[2]*0.75;
      basisvalues[5] = X*basisvalues[4]*1.8 - basisvalues[3]*0.8;
      for (unsigned int r = 0; r < 6; r++)
      {
        basisvalues[r] *= std::sqrt((0.5 + r));
      } // end loop over 'r'
      
      // Table(s) of coefficients
      static const double coefficients0[6] = \
      {0.245523187911996, 0.182253701099939, -0.345089824026312, 0.43306219754328, 0.175373705651426, -0.440643014954552};
      
      // Compute value(s)
      for (unsigned int r = 0; r < 6; r++)
      {
        *values += coefficients0[r]*basisvalues[r];
      } // end loop over 'r'
        break;
      }
    case 4:
      {
        
      // Array of basisvalues
      double basisvalues[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      
      // Declare helper variables
      
      // Compute basisvalues
      basisvalues[0] = 1.0;
      basisvalues[1] = X;
      basisvalues[2] = X*basisvalues[1]*1.5 - basisvalues[0]*0.5;
      basisvalues[3] = X*basisvalues[2]*1.66666666666667 - basisvalues[1]*0.666666666666667;
      basisvalues[4] = X*basisvalues[3]*1.75 - basisvalues[2]*0.75;
      basisvalues[5] = X*basisvalues[4]*1.8 - basisvalues[3]*0.8;
      for (unsigned int r = 0; r < 6; r++)
      {
        basisvalues[r] *= std::sqrt((0.5 + r));
      } // end loop over 'r'
      
      // Table(s) of coefficients
      static const double coefficients0[6] = \
      {0.245523187911996, -0.182253701099939, -0.345089824026311, -0.43306219754328, 0.175373705651425, 0.440643014954552};
      
      // Compute value(s)
      for (unsigned int r = 0; r < 6; r++)
      {
        *values += coefficients0[r]*basisvalues[r];
      } // end loop over 'r'
        break;
      }
    case 5:
      {
        
      // Array of basisvalues
      double basisvalues[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      
      // Declare helper variables
      
      // Compute basisvalues
      basisvalues[0] = 1.0;
      basisvalues[1] = X;
      basisvalues[2] = X*basisvalues[1]*1.5 - basisvalues[0]*0.5;
      basisvalues[3] = X*basisvalues[2]*1.66666666666667 - basisvalues[1]*0.666666666666667;
      basisvalues[4] = X*basisvalues[3]*1.75 - basisvalues[2]*0.75;
      basisvalues[5] = X*basisvalues[4]*1.8 - basisvalues[3]*0.8;
      for (unsigned int r = 0; r < 6; r++)
      {
        basisvalues[r] *= std::sqrt((0.5 + r));
      } // end loop over 'r'
      
      // Table(s) of coefficients
      static const double coefficients0[6] = \
      {0.368284781867994, 0.516385486449827, 0.188230813105261, -0.0618660282204685, -0.263060558477138, -0.220321507477276};
      
      // Compute value(s)
      for (unsigned int r = 0; r < 6; r++)
      {
        *values += coefficients0[r]*basisvalues[r];
      } // end loop over 'r'
        break;
      }
    }
    
  }

  /// Evaluate basis function i at given point x in cell (non-static member function)
  virtual void evaluate_basis(std::size_t i,
                              double* values,
                              const double* x,
                              const double* vertex_coordinates,
                              int cell_orientation) const
  {
    _evaluate_basis(i, values, x, vertex_coordinates, cell_orientation);
  }

  /// Evaluate all basis functions at given point x in cell (actual implementation)
  static void _evaluate_basis_all(double* values,
                                  const double* x,
                                  const double* vertex_coordinates,
                                  int cell_orientation)
  {
    // Helper variable to hold values of a single dof.
    double dof_values = 0.0;
    
    // Loop dofs and call evaluate_basis
    for (unsigned int r = 0; r < 6; r++)
    {
      _evaluate_basis(r, &dof_values, x, vertex_coordinates, cell_orientation);
      values[r] = dof_values;
    } // end loop over 'r'
  }

  /// Evaluate all basis functions at given point x in cell (non-static member function)
  virtual void evaluate_basis_all(double* values,
                                  const double* x,
                                  const double* vertex_coordinates,
                                  int cell_orientation) const
  {
    _evaluate_basis_all(values, x, vertex_coordinates, cell_orientation);
  }

  /// Evaluate order n derivatives of basis function i at given point x in cell (actual implementation)
  static void _evaluate_basis_derivatives(std::size_t i,
                                          std::size_t n,
                                          double* values,
                                          const double* x,
                                          const double* vertex_coordinates,
                                          int cell_orientation)
  {
throw std::runtime_error("// Function evaluate_basis_derivatives not generated (compiled with -fno-evaluate_basis_derivatives)");
  }

  /// Evaluate order n derivatives of basis function i at given point x in cell (non-static member function)
  virtual void evaluate_basis_derivatives(std::size_t i,
                                          std::size_t n,
                                          double* values,
                                          const double* x,
                                          const double* vertex_coordinates,
                                          int cell_orientation) const
  {
    _evaluate_basis_derivatives(i, n, values, x, vertex_coordinates, cell_orientation);
  }

  /// Evaluate order n derivatives of all basis functions at given point x in cell (actual implementation)
  static void _evaluate_basis_derivatives_all(std::size_t n,
                                              double* values,
                                              const double* x,
                                              const double* vertex_coordinates,
                                              int cell_orientation)
  {
    // Call evaluate_basis_all if order of derivatives is equal to zero.
    if (n == 0)
    {
      _evaluate_basis_all(values, x, vertex_coordinates, cell_orientation);
      return ;
    }
    
    // Compute number of derivatives.
    unsigned int num_derivatives = 1;
    for (unsigned int r = 0; r < n; r++)
    {
      num_derivatives *= 1;
    } // end loop over 'r'
    
    // Set values equal to zero.
    for (unsigned int r = 0; r < 6; r++)
    {
      for (unsigned int s = 0; s < num_derivatives; s++)
      {
        values[r*num_derivatives + s] = 0.0;
      } // end loop over 's'
    } // end loop over 'r'
    
    // If order of derivatives is greater than the maximum polynomial degree, return zeros.
    if (n > 5)
    {
      return ;
    }
    
    // Helper variable to hold values of a single dof.
    double dof_values[1];
    for (unsigned int r = 0; r < 1; r++)
    {
      dof_values[r] = 0.0;
    } // end loop over 'r'
    
    // Loop dofs and call evaluate_basis_derivatives.
    for (unsigned int r = 0; r < 6; r++)
    {
      _evaluate_basis_derivatives(r, n, dof_values, x, vertex_coordinates, cell_orientation);
      for (unsigned int s = 0; s < num_derivatives; s++)
      {
        values[r*num_derivatives + s] = dof_values[s];
      } // end loop over 's'
    } // end loop over 'r'
  }

  /// Evaluate order n derivatives of all basis functions at given point x in cell (non-static member function)
  virtual void evaluate_basis_derivatives_all(std::size_t n,
                                              double* values,
                                              const double* x,
                                              const double* vertex_coordinates,
                                              int cell_orientation) const
  {
    _evaluate_basis_derivatives_all(n, values, x, vertex_coordinates, cell_orientation);
  }

  /// Evaluate linear functional for dof i on the function f
  virtual double evaluate_dof(std::size_t i,
                              const ufc::function& f,
                              const double* vertex_coordinates,
                              int cell_orientation,
                              const ufc::cell& c) const
  {
    // Declare variables for result of evaluation
    double vals[1];
    
    // Declare variable for physical coordinates
    double y[1];
    switch (i)
    {
    case 0:
      {
        y[0] = vertex_coordinates[0];
      f.evaluate(vals, y, c);
      return vals[0];
        break;
      }
    case 1:
      {
        y[0] = vertex_coordinates[1];
      f.evaluate(vals, y, c);
      return vals[0];
        break;
      }
    case 2:
      {
        y[0] = 0.8*vertex_coordinates[0] + 0.2*vertex_coordinates[1];
      f.evaluate(vals, y, c);
      return vals[0];
        break;
      }
    case 3:
      {
        y[0] = 0.6*vertex_coordinates[0] + 0.4*vertex_coordinates[1];
      f.evaluate(vals, y, c);
      return vals[0];
        break;
      }
    case 4:
      {
        y[0] = 0.4*vertex_coordinates[0] + 0.6*vertex_coordinates[1];
      f.evaluate(vals, y, c);
      return vals[0];
        break;
      }
    case 5:
      {
        y[0] = 0.2*vertex_coordinates[0] + 0.8*vertex_coordinates[1];
      f.evaluate(vals, y, c);
      return vals[0];
        break;
      }
    }
    
    return 0.0;
  }

  /// Evaluate linear functionals for all dofs on the function f
  virtual void evaluate_dofs(double* values,
                             const ufc::function& f,
                             const double* vertex_coordinates,
                             int cell_orientation,
                             const ufc::cell& c) const
  {
    // Declare variables for result of evaluation
    double vals[1];
    
    // Declare variable for physical coordinates
    double y[1];
    y[0] = vertex_coordinates[0];
    f.evaluate(vals, y, c);
    values[0] = vals[0];
    y[0] = vertex_coordinates[1];
    f.evaluate(vals, y, c);
    values[1] = vals[0];
    y[0] = 0.8*vertex_coordinates[0] + 0.2*vertex_coordinates[1];
    f.evaluate(vals, y, c);
    values[2] = vals[0];
    y[0] = 0.6*vertex_coordinates[0] + 0.4*vertex_coordinates[1];
    f.evaluate(vals, y, c);
    values[3] = vals[0];
    y[0] = 0.4*vertex_coordinates[0] + 0.6*vertex_coordinates[1];
    f.evaluate(vals, y, c);
    values[4] = vals[0];
    y[0] = 0.2*vertex_coordinates[0] + 0.8*vertex_coordinates[1];
    f.evaluate(vals, y, c);
    values[5] = vals[0];
  }

  /// Interpolate vertex values from dof values
  virtual void interpolate_vertex_values(double* vertex_values,
                                         const double* dof_values,
                                         const double* vertex_coordinates,
                                         int cell_orientation,
                                         const ufc::cell& c) const
  {
    // Evaluate function and change variables
    vertex_values[0] = dof_values[0];
    vertex_values[1] = dof_values[1];
  }

  /// Map coordinate xhat from reference cell to coordinate x in cell
  virtual void map_from_reference_cell(double* x,
                                       const double* xhat,
                                       const ufc::cell& c) const
  {
    throw std::runtime_error("map_from_reference_cell not yet implemented.");
  }

  /// Map from coordinate x in cell to coordinate xhat in reference cell
  virtual void map_to_reference_cell(double* xhat,
                                     const double* x,
                                     const ufc::cell& c) const
  {
    throw std::runtime_error("map_to_reference_cell not yet implemented.");
  }

  /// Return the number of sub elements (for a mixed element)
  virtual std::size_t num_sub_elements() const
  {
    return 0;
  }

  /// Create a new finite element for sub element i (for a mixed element)
  virtual ufc::finite_element* create_sub_element(std::size_t i) const
  {
    return 0;
  }

  /// Create a new class instance
  virtual ufc::finite_element* create() const
  {
    return new ffc_form_2b1be6aa77ecca616bc92d2b088a24d1b832efb5_finite_element_0();
  }

};

/// This class defines the interface for a local-to-global mapping of
/// degrees of freedom (dofs).

class ffc_form_2b1be6aa77ecca616bc92d2b088a24d1b832efb5_dofmap_0: public ufc::dofmap
{
public:

  /// Constructor
  ffc_form_2b1be6aa77ecca616bc92d2b088a24d1b832efb5_dofmap_0() : ufc::dofmap()
  {
    // Do nothing
  }

  /// Destructor
  virtual ~ffc_form_2b1be6aa77ecca616bc92d2b088a24d1b832efb5_dofmap_0()
  {
    // Do nothing
  }

  /// Return a string identifying the dofmap
  virtual const char* signature() const
  {
    return "FFC dofmap for FiniteElement('Lagrange', Domain(Cell('interval', 1)), 5, None)";
  }

  /// Return true iff mesh entities of topological dimension d are needed
  virtual bool needs_mesh_entities(std::size_t d) const
  {
    switch (d)
    {
    case 0:
      {
        return true;
        break;
      }
    case 1:
      {
        return true;
        break;
      }
    }
    
    return false;
  }

  /// Return the topological dimension of the associated cell shape
  virtual std::size_t topological_dimension() const
  {
    return 1;
  }

  /// Return the geometric dimension of the associated cell shape
  virtual std::size_t geometric_dimension() const
  {
    return 1;
  }

  /// Return the dimension of the global finite element function space
  virtual std::size_t global_dimension(const std::vector<std::size_t>&
                                       num_global_entities) const
  {
    return num_global_entities[0] + 4*num_global_entities[1];
  }

  /// Return the dimension of the local finite element function space for a cell
  virtual std::size_t local_dimension() const
  {
    return 6;
  }

  /// Return the number of dofs on each cell facet
  virtual std::size_t num_facet_dofs() const
  {
    return 1;
  }

  /// Return the number of dofs associated with each cell entity of dimension d
  virtual std::size_t num_entity_dofs(std::size_t d) const
  {
    switch (d)
    {
    case 0:
      {
        return 1;
        break;
      }
    case 1:
      {
        return 4;
        break;
      }
    }
    
    return 0;
  }

  /// Tabulate the local-to-global mapping of dofs on a cell
  virtual void tabulate_dofs(std::size_t* dofs,
                             const std::vector<std::size_t>& num_global_entities,
                             const ufc::cell& c) const
  {
    unsigned int offset = 0;
    dofs[0] = offset + c.entity_indices[0][0];
    dofs[1] = offset + c.entity_indices[0][1];
    offset += num_global_entities[0];
    dofs[2] = offset + 4*c.entity_indices[1][0];
    dofs[3] = offset + 4*c.entity_indices[1][0] + 1;
    dofs[4] = offset + 4*c.entity_indices[1][0] + 2;
    dofs[5] = offset + 4*c.entity_indices[1][0] + 3;
    offset += 4*num_global_entities[1];
  }

  /// Tabulate the local-to-local mapping from facet dofs to cell dofs
  virtual void tabulate_facet_dofs(std::size_t* dofs,
                                   std::size_t facet) const
  {
    switch (facet)
    {
    case 0:
      {
        dofs[0] = 0;
        break;
      }
    case 1:
      {
        dofs[0] = 1;
        break;
      }
    }
    
  }

  /// Tabulate the local-to-local mapping of dofs on entity (d, i)
  virtual void tabulate_entity_dofs(std::size_t* dofs,
                                    std::size_t d, std::size_t i) const
  {
    if (d > 1)
    {
    throw std::runtime_error("d is larger than dimension (1)");
    }
    
    switch (d)
    {
    case 0:
      {
        if (i > 1)
      {
      throw std::runtime_error("i is larger than number of entities (1)");
      }
      
      switch (i)
      {
      case 0:
        {
          dofs[0] = 0;
          break;
        }
      case 1:
        {
          dofs[0] = 1;
          break;
        }
      }
      
        break;
      }
    case 1:
      {
        if (i > 0)
      {
      throw std::runtime_error("i is larger than number of entities (0)");
      }
      
      dofs[0] = 2;
      dofs[1] = 3;
      dofs[2] = 4;
      dofs[3] = 5;
        break;
      }
    }
    
  }

  /// Tabulate the coordinates of all dofs on a cell
  virtual void tabulate_coordinates(double** dof_coordinates,
                                    const double* vertex_coordinates) const
  {
    dof_coordinates[0][0] = vertex_coordinates[0];
    dof_coordinates[1][0] = vertex_coordinates[1];
    dof_coordinates[2][0] = 0.8*vertex_coordinates[0] + 0.2*vertex_coordinates[1];
    dof_coordinates[3][0] = 0.6*vertex_coordinates[0] + 0.4*vertex_coordinates[1];
    dof_coordinates[4][0] = 0.4*vertex_coordinates[0] + 0.6*vertex_coordinates[1];
    dof_coordinates[5][0] = 0.2*vertex_coordinates[0] + 0.8*vertex_coordinates[1];
  }

  /// Return the number of sub dofmaps (for a mixed element)
  virtual std::size_t num_sub_dofmaps() const
  {
    return 0;
  }

  /// Create a new dofmap for sub dofmap i (for a mixed element)
  virtual ufc::dofmap* create_sub_dofmap(std::size_t i) const
  {
    return 0;
  }

  /// Create a new class instance
  virtual ufc::dofmap* create() const
  {
    return new ffc_form_2b1be6aa77ecca616bc92d2b088a24d1b832efb5_dofmap_0();
  }

};

/// This class defines the interface for the tabulation of the cell
/// tensor corresponding to the local contribution to a form from
/// the integral over a cell.

class ffc_form_2b1be6aa77ecca616bc92d2b088a24d1b832efb5_cell_integral_0_otherwise: public ufc::cell_integral
{
public:

  /// Constructor
  ffc_form_2b1be6aa77ecca616bc92d2b088a24d1b832efb5_cell_integral_0_otherwise() : ufc::cell_integral()
  {
    // Do nothing
  }

  /// Destructor
  virtual ~ffc_form_2b1be6aa77ecca616bc92d2b088a24d1b832efb5_cell_integral_0_otherwise()
  {
    // Do nothing
  }

  /// Tabulate which form coefficients are used by this integral
  virtual const std::vector<bool> & enabled_coefficients() const
  {
    static const std::vector<bool> enabled({});
    return enabled;
  }

  /// Tabulate the tensor for the contribution from a local cell
  virtual void tabulate_tensor(double*  A,
                               const double * const *  w,
                               const double*  vertex_coordinates,
                               int cell_orientation) const
  {
    // Number of operations (multiply-add pairs) for Jacobian data:      3
    // Number of operations (multiply-add pairs) for geometry tensor:    0
    // Number of operations (multiply-add pairs) for tensor contraction: 3
    // Total number of operations (multiply-add pairs):                  6
    
    // Compute Jacobian
    double J[1];
    compute_jacobian_interval_1d(J, vertex_coordinates);
    
    // Compute Jacobian inverse and determinant
    double K[1];
    double detJ;
    compute_jacobian_inverse_interval_1d(K, detJ, J);
    
    // Set scale factor
    const double det = std::abs(detJ);
    
    // Compute geometry tensor
    const double G0_ = det;
    
    // Compute element tensor
    A[0] = 0.0659722222222221*G0_;
    A[1] = 0.0659722222222221*G0_;
    A[2] = 0.260416666666667*G0_;
    A[3] = 0.173611111111111*G0_;
    A[4] = 0.173611111111111*G0_;
    A[5] = 0.260416666666667*G0_;
  }

};

/// This class defines the interface for the assembly of the global
/// tensor corresponding to a form with r + n arguments, that is, a
/// mapping
///
///     a : V1 x V2 x ... Vr x W1 x W2 x ... x Wn -> R
///
/// with arguments v1, v2, ..., vr, w1, w2, ..., wn. The rank r
/// global tensor A is defined by
///
///     A = a(V1, V2, ..., Vr, w1, w2, ..., wn),
///
/// where each argument Vj represents the application to the
/// sequence of basis functions of Vj and w1, w2, ..., wn are given
/// fixed functions (coefficients).

class ffc_form_2b1be6aa77ecca616bc92d2b088a24d1b832efb5_form_0: public ufc::form
{
public:

  /// Constructor
  ffc_form_2b1be6aa77ecca616bc92d2b088a24d1b832efb5_form_0() : ufc::form()
  {
    // Do nothing
  }

  /// Destructor
  virtual ~ffc_form_2b1be6aa77ecca616bc92d2b088a24d1b832efb5_form_0()
  {
    // Do nothing
  }

  /// Return a string identifying the form
  virtual const char* signature() const
  {
    return "613e351f1a600c049ad85b634094511d5755ff9ddba4c56702ebfeafeb62e6adc7af51adb2ef704d206d5d2f6515eb88b155a1442efc4bf75719ef54981de910";
  }

  /// Return the rank of the global tensor (r)
  virtual std::size_t rank() const
  {
    return 1;
  }

  /// Return the number of coefficients (n)
  virtual std::size_t num_coefficients() const
  {
    return 0;
  }

  /// Return the number of cell domains
  virtual std::size_t num_cell_domains() const
  {
    return 0;
  }

  /// Return the number of exterior facet domains
  virtual std::size_t num_exterior_facet_domains() const
  {
    return 0;
  }

  /// Return the number of interior facet domains
  virtual std::size_t num_interior_facet_domains() const
  {
    return 0;
  }

  /// Return the number of point domains
  virtual std::size_t num_point_domains() const
  {
    return 0;
  }

  /// Return the number of custom domains
  virtual std::size_t num_custom_domains() const
  {
    return 0;
  }

  /// Return whether the form has any cell integrals
  virtual bool has_cell_integrals() const
  {
    return true;
  }

  /// Return whether the form has any exterior facet integrals
  virtual bool has_exterior_facet_integrals() const
  {
    return false;
  }

  /// Return whether the form has any interior facet integrals
  virtual bool has_interior_facet_integrals() const
  {
    return false;
  }

  /// Return whether the form has any point integrals
  virtual bool has_point_integrals() const
  {
    return false;
  }

  /// Return whether the form has any custom integrals
  virtual bool has_custom_integrals() const
  {
    return false;
  }

  /// Create a new finite element for argument function i
  virtual ufc::finite_element* create_finite_element(std::size_t i) const
  {
    switch (i)
    {
    case 0:
      {
        return new ffc_form_2b1be6aa77ecca616bc92d2b088a24d1b832efb5_finite_element_0();
        break;
      }
    }
    
    return 0;
  }

  /// Create a new dofmap for argument function i
  virtual ufc::dofmap* create_dofmap(std::size_t i) const
  {
    switch (i)
    {
    case 0:
      {
        return new ffc_form_2b1be6aa77ecca616bc92d2b088a24d1b832efb5_dofmap_0();
        break;
      }
    }
    
    return 0;
  }

  /// Create a new cell integral on sub domain i
  virtual ufc::cell_integral* create_cell_integral(std::size_t i) const
  {
    return 0;
  }

  /// Create a new exterior facet integral on sub domain i
  virtual ufc::exterior_facet_integral* create_exterior_facet_integral(std::size_t i) const
  {
    return 0;
  }

  /// Create a new interior facet integral on sub domain i
  virtual ufc::interior_facet_integral* create_interior_facet_integral(std::size_t i) const
  {
    return 0;
  }

  /// Create a new point integral on sub domain i
  virtual ufc::point_integral* create_point_integral(std::size_t i) const
  {
    return 0;
  }

  /// Create a new custom integral on sub domain i
  virtual ufc::custom_integral* create_custom_integral(std::size_t i) const
  {
    return 0;
  }

  /// Create a new cell integral on everywhere else
  virtual ufc::cell_integral* create_default_cell_integral() const
  {
    return new ffc_form_2b1be6aa77ecca616bc92d2b088a24d1b832efb5_cell_integral_0_otherwise();
  }

  /// Create a new exterior facet integral on everywhere else
  virtual ufc::exterior_facet_integral* create_default_exterior_facet_integral() const
  {
    return 0;
  }

  /// Create a new interior facet integral on everywhere else
  virtual ufc::interior_facet_integral* create_default_interior_facet_integral() const
  {
    return 0;
  }

  /// Create a new point integral on everywhere else
  virtual ufc::point_integral* create_default_point_integral() const
  {
    return 0;
  }

  /// Create a new custom integral on everywhere else
  virtual ufc::custom_integral* create_default_custom_integral() const
  {
    return 0;
  }

};

#endif
